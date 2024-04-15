clearvars,clc,clf
tic




%%%%%%% USER INPUTS %%%%%%%%%%

% PDE constants
% like, we could change these, but let's keep them as this unless we have a good reason
R = 1;
A = 1;
c = 1;
w = 1;

start_time = 0;
end_time = 1;
num_t = 1;
num_r = 50;
num_theta = 50;

make_images = true;
make_video = false;

%%%%%%% END USER INPUTS %%%%%%%%%%

% zeros of first order bessel function of the first kind
z1m = [3.83171, 7.01559, 10.1735, 13.3237, 16.4706, 19.6159, 22.7601, 25.9037, 29.0468, 32.1897, 35.3323, 38.4748, 41.6171, 44.7593, 47.9015, 51.0435, 54.1856, 57.3275, 60.4695, 63.6114, 66.7532, 69.8951, 73.0369, 76.1787, 79.3205, 82.4623, 85.604, 88.7458, 91.8875, 95.0292, 98.171, 101.313, 104.454, 107.596, 110.738, 113.879, 117.021, 120.163, 123.304, 126.446, 129.588, 132.729, 135.871, 139.013, 142.154, 145.296, 148.438, 151.579, 154.721, 157.863, 161.004, 164.146, 167.288, 170.429, 173.571, 176.712, 179.854, 182.996, 186.137, 189.279, 192.421, 195.562, 198.704];


% setting up geometry and time
dt = end_time/num_t;
dr = R/(num_r);
dtheta = 2 * pi / num_theta;

times = 0:dt:end_time;
r = 0:dr:R;
theta = 0:dtheta:2*pi;


Z_3D = zeros(length(r), length(theta), length(times));

% solving for all timesteps
for i = 1:length(times)
    time = times(i);
    Z_3D(:,:,i) = phi_all(r,theta,time,R,A,c,w,z1m);


    [R_grid, Theta_grid] = meshgrid(linspace(0, R, num_r+1), linspace(0, 2 * pi, num_theta+1));
    X = R_grid .* cos(Theta_grid);
    Y = R_grid .* sin(Theta_grid);
    contourf(X, Y, Z_3D(:,:,1), 50)
    colormap('parula');
    colorbar;
    % plotting and saving images
    if make_images
        X = r .* cos(theta);
        Y = r .* sin(theta)';
        figure('Renderer','zbuffer')
        hSurf = surf(X,Y,Z_3D(:,:,i));
        axis tight;    %# fix axis limits
        xlabel('x')
        ylabel('y')
        zlabel('z')
        zlim([-6 6])
        exportgraphics(gcf,"./images/" + string(i) + '.png','Resolution',600)
    end
end

% saving the data
num_digits = 20;
writematrix(round(Z_3D, num_digits), './data/Z_3D_data.txt', 'WriteMode', 'append', 'Delimiter', 'tab');
writematrix(round(r, num_digits), './data/r_data.txt', 'WriteMode', 'append', 'Delimiter', 'tab');
writematrix(round(theta, num_digits), './data/theta_data.txt', 'WriteMode', 'append', 'Delimiter', 'tab');
writematrix(round(times, num_digits), './data/times_data.txt', 'WriteMode', 'append', 'Delimiter', 'tab');


%%%%%%%%% video %%%%%%%%%%%%%%%%%%
if make_video
    figure('Renderer','opengl')

    %# this is the surface we will be animating
    Z0 = Z_3D(:,:,1);
    hSurf = surf(X,Y,Z0);
    axis tight;    %# fix axis limits

    %# these are some fixed lines
    %hLine(1) = line([0 50], [0 50], [-5 5], 'Color','r' ,'LineWidth',4);
    %hLine(2) = line([40 0], [0 40], [-5 5], 'Color','g' ,'LineWidth',4);

    %# some text as well
    %hTxt = text(10,40,5, '0');
    zlim([-6 6]);

    %% Initialize video
    myVideo = VideoWriter('video_of_analytical_sol'); %open video file
    myVideo.FrameRate = 10;  %can adjust this, 5 - 10 works well for me
    open(myVideo)


    %# iterations
    for j = 1:625
        if mod(j,10) == 0
            string(j)
        end
        %# animate the Z-coordinates of the surface
        set(hSurf, 'ZData',Z_3D(:,:,j))
        set(hSurf, 'EdgeAlpha',.1)
        %set(hSurf, 'FaceLighting','flat')
        %# change text
        %set(hTxt, 'String',num2str(j))

        %# flush + a small delay
        pause(0.01)


        cdata = print('-RGBImage','-r600','-noui'); % increased dpi to 600 (beware)
        frame = im2frame(cdata);                    % convert image to frame
        writeVideo(myVideo,frame); 
    end

    close(myVideo)
end



toc


%%%%%%%%%%%%%%% functions that solve it %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function Z = phi_all(r, theta, t, R, A, c, w, z1m)

Z = zeros(length(r),length(theta));

    fprintf('Progress:     0%%');
    for i = 1:length(r)
        for j = 1:length(theta)
            Z(i,j) = phi_indv(r(i), theta(j), t, R, A, c, w, z1m);
        end
        progress = i / length(r) * 100;
        fprintf('\b\b\b\b%3.0f%%', progress);
    end
    
    fprintf('\n');
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