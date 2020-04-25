% Load the initial data
load data;

train.image = (train.image * 2) - 1;
test.image = (test.image * 2) - 1;
accs = zeros(30, 1);

% for r = 1 : 1
%     tic
%     results = svd_comp(train, test, r);
%     time(r) = toc;
%     accs(r) = check(results, test);
% end

show_images(train);
    
% figure(1)
% clf
% plot(1:length(accs),accs,'LineWidth',2);
% xlabel('r value')
% ylabel('accuracy')
% set(gca,'FontWeight','Bold','FontSize',16)
% print -dpng latex/images/svd/r_values.png
% 
% figure(2)
% clf
% plot(1:length(time),time,'LineWidth',2);
% xlabel('r value')
% ylabel('time in sec')
% set(gca,'FontWeight','Bold','FontSize',16)
% print -dpng latex/images/svd/time.png


function result = svd_comp(train, test, r)
    result = zeros(length(test.label), 1);

    A = cell(1, 10);
    U = cell(1, 10);
    S = cell(1, 10);
    V = cell(1, 10);
    
%     % subtract the average vector
%     mean_vec = mean(azip, 2);
%     azip = azip - mean_vec;
%     testzip = testzip - mean_vec;
    
    % create a matrix for each digit
    for i = 1 : length(train.label)
        A{train.label(i) + 1} = [A{train.label(i) + 1}, train.image(:, i)];
    end

    % decompose the matrix for each digit
    for i = 1:length(A)
        [U{i}, S{i}, V{i}] = svd(A{i});
    end

%     figure(1)
%     clf
%     show(U{4}(:,1));
%     print -dpng latex/images/svd/U1.png
%     
%     
%     figure(2)
%     clf
%     show(U{4}(:,2));
%     print -dpng latex/images/svd/U2.png
%     
%     figure(3)
%     clf
%     show(U{4}(:,3));
%     print -dpng latex/images/svd/U3.png
% 
    figure(4)
    clf
    show(test.image(:, 3));
    print -dpng latex/images/svd/test3.png

    figure(5)
    clf
    i = 4; j = 3;
    show(A{i} * inv(A{i}.' * A{i}) * A{i}.' * test.image(:, j));
    print -dpng latex/images/svd/test3_proj.png

    figure(6)
    clf
    show(test.image(:, 1));
    print -dpng latex/images/svd/test9.png
    
    figure(7)
    clf
    i = 4; j = 1;
    show(A{i} * inv(A{i}.' * A{i}) * A{i}.' * test.image(:, j));
    print -dpng latex/images/svd/test9_proj.png
    
    % find the minimum relative residual
    min_rel_res = zeros(1, length(A));
    for j = 1 : length(test.label)
        for i = 1:length(A)
            orth_proj = U{i}(:, 1:r) * transpose(U{i}(:, 1:r)) * test.image(:, j);
%             orth_proj = A{i} * inv(A{i}.' * A{i}) * A{i}.' * test.image(:, j);
            min_rel_res(i) = norm(test.image(:, j)- orth_proj);
        end

        % store index of the smallest norm
        [~, i] = min(min_rel_res);
        result(j) = i - 1;
    end
end


function result = find_zero_distance(train, test, to_image)
    result = [];
    count = 1;
    threshold = 0.1;
    
    for i = 1 : length(test.label)
        if td(test.image(:, to_image), test.image(:, i)) < threshold
            result(:, count) = test.image(:, i);
            count = count + 1;
        end
    end
    
    for i = 1 : length(train.label)
        if td(test.image(:, to_image), train.image(:, i)) < threshold
            result(:, count) = train.image(:, i);
            count = count + 1;
        end
    end
end

function show_images(train)
    small_six = train.image(:, 32);
    fat_six = train.image(:, 1);
    train_four = train.image(:, 3);
    
    difference_46 = (train_four - fat_six).^2;
    difference_66 = (small_six - fat_six).^2;
    difference_6left6 = (small_six - left_shift_six(small_six, 4)).^2;
    show(left_shift_six(small_six, 4));
%     show(difference_6left6);
    sqrt(sum(difference_46))
    sqrt(sum(difference_66))
    sqrt(sum(difference_6left6))
end

function left_six = left_shift_six(small_six, p)
    six_grid = zeros(16, 16);
    for i = 1 : 256
        six_grid(floor((i - 1) / 16) + 1, mod(i - 1, 16) + 1) = small_six(i);
    end
    for i = 1 : p
        left_shift();
    end
    left_six = zeros(256, 1);
    for i = 1 : 256
        left_six(i) = six_grid(floor((i - 1) / 16) + 1, mod(i - 1, 16) + 1);
    end
    
    function left_shift()
        for i = 1 : 15
            six_grid(:, i) = six_grid(:, i + 1);
        end
        six_grid(:,16) = zeros(16, 1);
    end
end

function result = neural_net_simple(train, test)
    result = zeros(length(test.label), 1);

    % One hot vector
    y = train.label == (0:9)';
    v = test.label == (0:9)';

    % Initialize weights and biases
    rng(5000);
    W2 = 0.5 * randn(length(train.image(:, 1)), 64)'; b2 = 0.5 * randn(64, 1);
    W3 = 0.5 * randn(64, 32)'; b3 = 0.5 * randn(32, 1); 
    W4 = 0.5 * randn(32, 10)'; b4 = 0.5 * randn(10, 1);
    
    % Forward and Back propagate 
    % learning rate
    eta = 0.3;
    % number of SG iterations 
    N = 3e4;
    stepsize = 100;
    max_counter = N / stepsize;
    eval_counter = 1;
    train_cost = zeros(max_counter, 1);
    eval_cost = zeros(max_counter, 1);
    train_acc = zeros(max_counter, 1);
    eval_acc = zeros(max_counter, 1);
    
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
        
        if mod(counter, stepsize) == 0
            [train_cost(eval_counter), eval_cost(eval_counter)] = cost(W2, W3, W4, b2, b3, b4);
            [train_acc(eval_counter), eval_acc(eval_counter)] = accuracy(W2, W3, W4, b2, b3, b4);
            eval_counter = eval_counter + 1;
        end
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
    
    figure(1)
    clf
    plot(1:stepsize:N,train_cost,'LineWidth',2);
    hold on;
    plot(1:stepsize:N,eval_cost,'LineWidth',2);
    legend('training', 'test');
    xlabel('Iteration Number')
    ylabel('Value of cost function')
    set(gca,'FontWeight','Bold','FontSize',16)
    print -dpng latex/images/nn/mse_data.png
    
    figure(2)
    clf
    plot(1:stepsize:N,train_acc,'LineWidth',2);
    hold on;
    plot(1:stepsize:N,eval_acc,'LineWidth',2);
    legend('training', 'test','Location','southeast');
    xlabel('Iteration Number')
    ylabel('Accuracy')
    set(gca,'FontWeight','Bold','FontSize',16)
    print -dpng latex/images/nn/acc_data.png
    
    function [train_cost, eval_cost] = cost(W2, W3, W4, b2, b3, b4)
         costvec = zeros(length(train.label), 1);
         for i = 1:length(train.label)
             x_test = train.image(:, i);
             a2 = activate(x_test, W2, b2);
             a3 = activate(a2, W3, b3);
             a4 = activate(a3, W4, b4);
             costvec(i) = norm(y(:, i) - a4, 2);
         end
         train_cost = norm(costvec, 2)^2;
         
         costvec = zeros(length(test.label), 1);
         for i = 1:length(test.label)
             x_test = test.image(:, i);
             a2 = activate(x_test, W2, b2);
             a3 = activate(a2, W3, b3);
             a4 = activate(a3, W4, b4);
             costvec(i) = norm(v(:, i) - a4, 2);
         end
         eval_cost = norm(costvec, 2)^2;
    end

    function [train_acc, eval_acc] = accuracy(W2, W3, W4, b2, b3, b4)
        success = 0;
        for i = 1 : length(test.label)
            x_test = test.image(:, i);

            % Forward pass
            a2 = activate(x_test, W2, b2);
            a3 = activate(a2, W3, b3);
            a4 = activate(a3, W4, b4);

            % Check whether the neuron with the highest value is correct
            [~, j] = max(a4);
            if (j - 1 == test.label(i))
                success = success + 1;
            end
        end
        eval_acc = success / i;
        
        success = 0;
        for i = 1 : length(train.label)
            x_test = train.image(:, i);

            % Forward pass
            a2 = activate(x_test, W2, b2);
            a3 = activate(a2, W3, b3);
            a4 = activate(a3, W4, b4);

            % Check whether the neuron with the highest value is correct
            [~, j] = max(a4);
            if (j - 1 == train.label(i))
                success = success + 1;
            end
        end
        train_acc = success / i;
    end
end

function successrate = check(result, test)
    success = zeros(10, 1);
    
    for i = 1 : length(test.label)
        if test.label(i) == result(i)
            success(test.label(i) + 1) = success(test.label(i) + 1) + 1;
        end
    end

    % Calculate success rate
    successrate = sum(success) / i;
end