K = 5;
P = [ 0.1, 0.325, 0.25, 0.325; 
      0.4, 0,     0.4,  0.2;
      0.2, 0.2,   0.2,  0.4;
      1,   0,     0,    0];

% the cost function log
C = -log(P) 
%C = log(1-P);
%C = 1-P;

% define Vk 2D array and init to inf
Vk = inf(3,K);
letters_group_size = size(P,1)-1;

% update for the last K-1 time index (next is '-')
for i=1:letters_group_size
    Vk(i,K) = C(i,letters_group_size+1);
end

% we go back using the bellman recursive equation
% note that we allow in the backword direction also to start
% with any character. for the HW we would start only from 'B'
% when we get the most probable word.
for k=K-1:-1:1
    
    % states, actions and next state function for 
    % the middle time indexes
    Sk = [1:letters_group_size];
    Ak = [1:letters_group_size];
    Fk = Sk;
    % go through the states at that time index
    for i=Sk(1):Sk(end)
        costs = C(i,Ak)+ Vk(Fk,k+1)';
        Vk(i,k) = min(costs);
    end
end

%building the most probable word
path = zeros(1,K);

% we start only w/ 'B'
B_index = 1;
path(1) = B_index;

prev_index = B_index;
for k=1:K-1
    costs = C(prev_index, Ak)+ Vk(Fk,k+1)';
    [value, index] = min(costs);
    path(k+1) = index;
    prev_index = index;
end

lut = [char('B'), char('K'), char('O')];
word = lut(path);
string(word)