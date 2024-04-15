clearvars,clf
load("../data/all_data.mat")



resolution_dpi = 300;
redo_2d = true;
redo_3d = true;





if redo_2d
    parfor i = 1:length(times)
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
    end
end

if redo_3d
    parfor i = 1:length(times)
        figure('Renderer','zbuffer')
        hSurf = surf(X,Y,Z_3D(:,:,i));
        set(hSurf, 'EdgeAlpha',0.05)
        axis tight;    %# fix axis limits
        zlim([-6 6])
        title("time = " + string(times(i)))
        exportgraphics(gcf,"../images/3D" + string(i) + '_3D.png','Resolution',resolution_dpi)
    end
end

