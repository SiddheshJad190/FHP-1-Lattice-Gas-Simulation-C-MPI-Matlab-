function [Q, R] = qrfactor(A)
    % Householder QR factorization
    [m, n] = size(A);
    Q = eye(m);
    R = A;

    for k = 1:n
        x = R(k:m, k);
        e1 = zeros(length(x), 1); e1(1) = 1;
        v = sign(x(1)) * norm(x) * e1 + x;
        v = v / norm(v);
        Hk = eye(m);
        Hk(k:m, k:m) = Hk(k:m, k:m) - 2 * (v * v');
        R = Hk * R;
        Q = Q * Hk';
    end
end
