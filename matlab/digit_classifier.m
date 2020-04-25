% Load the initial data.
% For this to work you need to have a file data.mat in which
% there are structs train and test containing images and labels
load data;

% get result from the chosen classifier
% svd_comp can be replaced by one of the below defined functions
% only the name has to be changed
tic
result = svd_comp(train, test, 15, false);
toc

% Finally, check the result against the test labels and display the
% accuracy
check(result, test)

function result = svd_comp(train, test, r, orthonormal)
    train.image = (train.image * 2) - 1;
    test.image = (test.image * 2) - 1;
    
    result = zeros(length(test.label), 1);

    A = cell(1, 10);
    U = cell(1, 10);
    S = cell(1, 10);
    V = cell(1, 10);
      
    % create a matrix for each digit
    for i = 1 : length(train.label)
        A{train.label(i) + 1} = [A{train.label(i) + 1}, ...
            train.image(:, i)];
    end

    % decompose the matrix for each digit
    for i = 1:length(A)
        [U{i}, S{i}, V{i}] = svd(A{i});
    end

    % find the minimum relative residual
    min_rel_res = zeros(1, length(A));
    for j = 1 : length(test.label)
        for i = 1:length(A)
            if (orthonormal)
                orth_proj = U{i}(:, 1:r) * ...
                    transpose(U{i}(:, 1:r)) * test.image(:, j);
            else
                orth_proj = U{i}(:, 1:r) * ...
                    inv(U{i}(:, 1:r)' * U{i}(:, 1:r)) * ...
                    U{i}(:, 1:r)' * test.image(:, j);
            end
            min_rel_res(i) = norm(test.image(:, j)- orth_proj);
        end

        % store index of the smallest norm
        [~, i] = min(min_rel_res);
        result(j) = i - 1;
    end
end

function result = neural_net_simple(train, test)
    result = zeros(length(test.label), 1);

    % One hot vector
    y = train.label == (0:9)';

    % Initialize weights and biases
    rng(5000);
    W2 = 0.5 * randn(length(train.image(:, 1)), 64)'; 
    b2 = 0.5 * randn(64, 1);
    W3 = 0.5 * randn(64, 32)'; 
    b3 = 0.5 * randn(32, 1); 
    W4 = 0.5 * randn(32, 10)'; 
    b4 = 0.5 * randn(10, 1);
    
    % Forward and Back propagate 
    % learning rate
    eta = 0.3;
    % number of SG iterations 
    N = 3e4;

    for counter = 1:N
        % choose a training point at random
        k = randi(length(train.label));
        x_train = train.image(:, k);
        % Forward pass
        a2 = activate(x_train, W2, b2);
        a3 = activate(a2, W3, b3);
        a4 = activate(a3, W4, b4);
        % Backward pass
        delta4 = a4 .* (1 - a4) .* (a4 - y(:, k));
        delta3 = a3 .* (1 - a3) .* (W4' * delta4);
        delta2 = a2 .* (1 - a2) .* (W3' * delta3);
        % Gradient step
        W2 = W2 - eta * delta2 * x_train';
        W3 = W3 - eta * delta3 * a2';
        W4 = W4 - eta * delta4 * a3';
        b2 = b2 - eta * delta2;
        b3 = b3 - eta * delta3;
        b4 = b4 - eta * delta4;
    end

    for i = 1:length(test.label)
        x_test = test.image(:, i);
        
        % Forward pass
        a2 = activate(x_test, W2, b2);
        a3 = activate(a2, W3, b3);
        a4 = activate(a3, W4, b4);
        
        % Check whether the neuron with the highest value is correct
        [~, j] = max(a4);
        result(i) = j - 1;
    end
end

function result = nearest_neighbour(train, test)
    k = 1;
    model = fitcknn(transpose(train.image), transpose(train.label));
    model.NumNeighbors = k;
    result = predict(model, transpose(test.image));
end

function result = nearest_neighbour_tangent(train, test, k)
    result = zeros(length(test.label), 1);
    % For each testcase
    for i=1:length(test.label)
        best_norm = Inf;
        best_labels = -ones(k, 1);
        
        % Determine the best norm
        for j=1:length(train.label)
            current_norm = norm(train.image(:,j)- test.image(:,i));
            if (current_norm <= best_norm)
                best_norm = current_norm;
                
                for p = 2 : k
                    best_labels(p) = best_labels(p - 1);
                end
                best_labels(1) = train.label(j);
            end
        end
        result(i) = best_labels(1);
    end
end

function result = nearest_neighbour_mean(train, test)
    % average vector 1 contains the average vector of the digit 0
    % create an average vector for each digit
    average_vectors = zeros(256, 10);
    number_of_vectors = zeros(1, 10);
    result = zeros(length(test.label), 1);

    % compute the total vector for each digit
    for i = 1 : length(train.label)
        digit = train.label(i) + 1;
        average_vectors(:, digit) = average_vectors(:, digit) ...
            + train.image(:, i);
        number_of_vectors(digit) = number_of_vectors(digit) + 1;
    end

    % normalise the vectors
    for i = 1 : 10
        average_vectors(:, i) = average_vectors(:, i) ... 
            / number_of_vectors(i);
    end
    
    % check which average vector is closest with respect to the dist
    for i = 1 : length(test.label)
        best_norm = Inf;
        best_guess = -1;

        % Determine the best norm
        for j = 1 : 10
            current_norm = sum(abs((average_vectors(:, j) ... 
                - test.image(:, i))));
            if (current_norm < best_norm)
                best_norm = current_norm;
                best_guess = j - 1;
            end
        end
        result(i) = best_guess;
    end
end

function successrate = check(result, test)
    success = zeros(10, 1);
    
    for i = 1 : length(test.label)
        if test.label(i) == result(i)
            success(test.label(i) + 1) ... 
                = success(test.label(i) + 1) + 1;
        end
    end

    % Calculate success rate
    successrate = sum(success) / i;
end
