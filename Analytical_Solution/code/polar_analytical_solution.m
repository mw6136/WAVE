clearvars,clf,clc
tic




%%%%%%% USER INPUTS %%%%%%%%%%

start_time = 0;
end_time = 0.5;
number_of_timesteps = 51;

make_images = true;

%%%%%%% END USER INPUTS %%%%%%%%%%

N = 200;

rs_linspace = linspace(0,1,N);
thetas_linspace = linspace(0,2*pi,N);

[thetas,rs] = meshgrid(thetas_linspace,rs_linspace);

% Deleting previous results, be careful
% useful if you create fewer images this time than previously
% also the data might just append to the text files?
delete ../data/*
delete ../images/2D/*
delete ../images/3D/*



% PDE  (and other) constants
% like, we could change these, but let's keep them as this unless we have a good reason
R = 1;
A = 1;
c = 1;
w = 1;

resolution_dpi = 300;


% zeros of first order bessel function of the first kind
z1m = [3.83171, 7.01559, 10.1735, 13.3237, 16.4706, 19.6159, 22.7601, 25.9037, 29.0468, 32.1897, 35.3323, 38.4748, 41.6171, 44.7593, 47.9015, 51.0435, 54.1856, 57.3275, 60.4695, 63.6114, 66.7532, 69.8951, 73.0369, 76.1787, 79.3205, 82.4623, 85.604, 88.7458, 91.8875, 95.0292, 98.171, 101.313, 104.454, 107.596, 110.738, 113.879, 117.021, 120.163, 123.304, 126.446, 129.588, 132.729, 135.871, 139.013, 142.154, 145.296, 148.438, 151.579, 154.721, 157.863, 161.004, 164.146, 167.288, 170.429, 173.571, 176.712, 179.854, 182.996, 186.137, 189.279, 192.421, 195.562, 198.704];


% setting up geometry and time
times = linspace(start_time,end_time,number_of_timesteps);
Z_3D = zeros(length(rs), length(thetas), length(times));

% initializing X and Y
[X,Y] = pol2cart(thetas,rs);

% solving for all timesteps
parfor i = 1:length(times)
    time = times(i);
    Z_3D(:,:,i) = phi_all(rs,thetas,time,R,A,c,w,z1m);
    %save('../data/data' + string(i) + '.mat','-formstruct',s)
    % https://www.mathworks.com/help/parallel-computing/save-variables-in-parfor-loop.html

    % plotting and saving images
    if make_images
        % 2D
        figure()
        h = pcolor(X,Y,Z_3D(:,:,i));
        set(h, 'EdgeColor', 'none');
        axis square
        colorbar();
        caxis([-6 6])
        xlabel("X")
        ylabel("Y")
        title("time = " + string(times(i)))
        set(gca,'FontSize',13)
        exportgraphics(gcf,"../images/2D/" + string(i) + '_2D.png','Resolution',resolution_dpi)

        % 3D
        figure('Renderer','zbuffer')
        hSurf = surf(X,Y,Z_3D(:,:,i));
        set(hSurf, 'EdgeAlpha',0.05)
        axis tight;    %# fix axis limits
        zlim([-6 6])
        title("time = " + string(times(i)))
        exportgraphics(gcf,"../images/3D/" + string(i) + '_3D.png','Resolution',resolution_dpi)
    end
end

% saving the data
num_digits = 20;
%writematrix(round(Z_3D, num_digits), '../data/Z_3D_data.txt', 'WriteMode', 'append', 'Delimiter', 'tab');
writematrix(round(rs, num_digits), '../data/r_data.txt', 'WriteMode', 'append', 'Delimiter', 'tab');
writematrix(round(thetas, num_digits), '../data/theta_data.txt', 'WriteMode', 'append', 'Delimiter', 'tab');
writematrix(round(times, num_digits), '../data/times.txt', 'WriteMode', 'append', 'Delimiter', 'tab');

for i = 1:length(times)
    Z = Z_3D(:,:,i);
    writematrix(round(Z, num_digits), '../data/Z' + string(i) + '.txt', 'WriteMode', 'append', 'Delimiter', 'tab');
end

%%%%%%%%% video %%%%%%%%%%%%%%%%%%



save('../data/all_data.mat')

toc


%%%%%%%%%%%%%%% functions that solve it %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function Z = phi_all(rs, thetas, t, R, A, c, w, z1m)

total_elem = length(thetas)^2;
Z = zeros(size(thetas));

    % redefine z for new t
    parfor i = 1:total_elem
        r = rs(i);
        theta = thetas(i);
        if r > R
            Z(i) = nan;
        else
            Z(i) = phi_indv(r, theta, t, R, A, c, w, z1m);
        end
    end

end


function phi = phi_indv(r, theta, t, R, A, c, w, z1m)

M = 1:length(z1m);
phi_tilde = zeros(size(M));

for m = M
    lam1m = (z1m(m)/R)^2;

    T1 = 2*A / ((c^2*lam1m - w^2)*R^2);

    T2top_integrand = @(eta) (w.^2.*eta.^2 + 3*c.^2).*eta .* besselj(1,sqrt(lam1m)*R.*eta);
    T2top = integral(T2top_integrand,0,1);
    T2bot = (besselj(2,sqrt(lam1m)*R))^2;
    T2 = T2top / T2bot;

    T3 = 2*A / T2bot;

    T4integrand = @(eta) eta.^3 .* besselj(1,sqrt(lam1m)*R*eta);
    T4 = integral(T4integrand,0,1);

    T5 = 1 / ((c^2*lam1m - w^2)*R^2);

    T6 = T2top;

    T7 = cos(c*t*sqrt(lam1m));

    T8 = cos(theta) * besselj(1,sqrt(lam1m)*r);

    phi_tilde(m) = (T1 * T2 * cos(w*t) - T3 * (T4 + T5 * T6) * T7) * T8;
end
phi_tilde = sum(phi_tilde);
phi = A * (r/R)^2 * cos(w*t) * cos(theta) + phi_tilde;

end
