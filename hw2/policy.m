P = [ 0,        0.3125, 0.6875; 
      0.58333,  0,      0.41666;
      0.75,     0.25,   0];

P2 = P*P;
P3 = P*P2;

P_a1 = [    0,      0.5,    0.5; 
            2/3,    0,      1/3;
            0.75,   0.25,   0];

P_a2 = [    0,      0.125,  0.875; 
            0.5,    0,      0.5;
            0.75,   0.25,   0];

T=3;
V = zeros(3,T+1);
VT = [ 0.7; 1; 0.5];
R_a1 = [ 0.2; 1; 0.5];
R_a2 = [ 0.7; 0; 0.5];
PiT = [ 2; 1; 2];
V(:,T+1) = VT;
Pi(:,T+1) = PiT;
for t=T:-1:1
    a1_val = R_a1 + P_a1 * V(:,t+1);
    a2_val = R_a2 + P_a2 * V(:,t+1);
    A_val(:,1) = a1_val;
    A_val(:,2) = a2_val;
    [V(:,t), Pi(:,t)] = max(A_val,[],2);
end

V
Pi


