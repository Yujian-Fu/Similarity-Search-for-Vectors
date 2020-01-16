function Compute_LID_with_Distance(file_name, dataset_name, save_path)

size_of_point = 8;
color_of_point = 'r';

%Distance_feature = textread(file_name);
Distance_feature = file_name;
[instances, length] = size(Distance_feature);

LID_MLE = zeros([1, instances]);
LID_RV = zeros([1, instances]);
means_MLE = zeros([1, 39]);
means_RV = zeros([1, 39]);
means_index = 0;
k_Estimators = linspace(20, 780, 39);

for k_Estimator = k_Estimators
    means_index = means_index +1;
    for count = 1:instances
       IDx = 0;
       Distance = Distance_feature(count, :);
       %sort and eliminate the 0 in distance
       Distance = sort(Distance);
       j = 1;
       while (Distance(j) == 0)
           j = j+1;
       end
       Distance = Distance(j:end);
       %use MLE estimator
       for dis_index = 1:k_Estimator
           IDx = IDx + (1/k_Estimator)*log(Distance(dis_index)/Distance(k_Estimator));
       end
       IDx = -1/(IDx);
       LID_MLE(count) = IDx;
       %use RV estimator, all variable is the same within the paper
       numerator = log(k_Estimator)-log(fix(k_Estimator/2));
       denominator = log(Distance(k_Estimator))-log(Distance(fix(k_Estimator/2)));
       IDrv = numerator/denominator;
       LID_RV(count) = IDrv;
    end

    %draw the figure with MLE
    LID_mean = mean(LID_MLE);
    LID_std = std(LID_MLE);
    LID_median = median(LID_MLE);
    means_MLE(means_index) = LID_mean;
    
    if k_Estimator == 780
        figure
        x = linspace(1, instances, instances);
        scatter(x, LID_MLE, size_of_point, color_of_point, 'filled');

        xticks([0, 200, 400, 600, 800]);
        %xticklabels({'0', '2K', '4K', '6K', '8K', '10K'});

        xlabel('number of instances');
        ylabel('LID MLE');
        txt = {[' mean ', num2str(LID_mean), ' std ', num2str(LID_std)], [' median ', num2str(LID_median), ' k ', num2str(k_Estimator)]};
        yl = ylim;
        text(0, yl(2)*0.9, txt);
        title(['the LID distribution of ', dataset_name, ' with MLE ']);
        savefig(gcf, [save_path, 'LID MLE distribution ',  num2str(k_Estimator), '.fig']);
    end
    
    %draw the figure with MLE
    LID_mean = mean(LID_RV);
    LID_std = std(LID_RV);
    LID_median = median(LID_RV);
    means_RV(means_index) = LID_mean;

    if k_Estimator == 780
        figure
        x = linspace(1, instances, instances);
        scatter(x, LID_RV, size_of_point, color_of_point, 'filled');

        xticks([0, 200, 400, 600, 800]);
%         xticklabels({'0', '2K', '4K', '6K', '8K', '10K'});

        xlabel('number of instances');
        ylabel('LID RV');
        txt = {[' mean ', num2str(LID_mean), ' std ', num2str(LID_std)], [' median ', num2str(LID_median), ' k ', num2str(k_Estimator)]};
        yl = ylim;
        text(0, yl(2)*0.9, txt);
        title(['the LID distribution of ', dataset_name, ' with RV ']);
        savefig([save_path, 'LID RV distribution ',  num2str(k_Estimator), '.fig']);
    end
end

figure 
plot(k_Estimators, means_MLE, k_Estimators, means_RV);
legend( "MLE", "RV");
xlabel("k value in LID");
ylabel("LID");
grid on
title("LID with different estimators");
savefig([save_path, 'LID estimation.fig']);


