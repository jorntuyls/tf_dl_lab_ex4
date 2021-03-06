function unt()

%clear all;
close all;
set(gcf,'color','w');

iscenario = 2;
colors = {'red','blue','green','black','magenta','cyan', ...
    [0.4 0.7 0.1],[0.7 0.4 0.1],[0.1 0.4 0.7],[0.7, 0.7, 0]};

if (iscenario == 1)
    
    for ialgo=[1, 2]
        nruns = 10;
        ybest_runs = [];
        for irun=1:nruns
            if (ialgo == 1)    filename = ['randomsearch_' num2str(irun-1) '.txt'];     end;
            if (ialgo == 2)    filename = ['hyperband_' num2str(irun-1) '.txt'];        end;
            M = dlmread(filename);
            xvals = M(:,1)
            if (ialgo == 1)    
                yvals = M(:,7)
            else
                yvals = M(:,8)
            end;
            ybest = yvals;  % already computed in our python code
            ybest_runs(irun,:) = ybest;
        end;
        ybest_median = mean(ybest_runs);
        semilogx(xvals, ybest_median, 'color', colors{ialgo}); hold on;
    end;
    ylim([0 1]);
    xlim([0 20]);
    legend({'Random','Hyperband'});
    xlabel('number of function evaluations','fontsize',16);
    ylabel('best validation loss','fontsize',16); 
    title('random search vs hyperband','fontsize',16);
end;

if (iscenario == 2)
    
    for ialgo=[1, 2]
        nruns = 10;
        ybest_runs = [];
        for irun=1:nruns
            if (ialgo == 1)    filename = ['randomsearch_' num2str(irun-1) '.txt'];     end;
            if (ialgo == 2)    filename = ['hyperband_' num2str(irun-1) '.txt'];        end;
            M = dlmread(filename);
            xvals = M(:,2)
            if (ialgo == 1)    
                yvals = M(:,7)
            else
                yvals = M(:,8)
            end;
            ybest = yvals;  % already computed in our python code
            ybest_runs(irun,:) = ybest;
        end;
        ybest_median = mean(ybest_runs);
        semilogx(xvals, ybest_median, 'color', colors{ialgo}); hold on;
    end;
    ylim([0 1]);
    xlim([0 10000]);
    legend({'Random','Hyperband'});
    xlabel('time','fontsize',16);
    ylabel('best validation loss','fontsize',16); 
    title('random search vs hyperband','fontsize',16);
end;
%export_fig('1.png', ['-r300']);