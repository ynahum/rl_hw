N = 12;
X = 800;

% initialize array of size N x X to zeroes and the
a = zeros(N,X);
a(1,:) = 1;

for i = 2:N 
    for j = 1:X
        if (i == j )
            a(i,j) = 1;
        elseif( i > j )
            a(i,j) = 0;
        else
            a(i,j) = a(i-1,j-1) + a(i,j-1);
        end
    end
end

a(N,X)
