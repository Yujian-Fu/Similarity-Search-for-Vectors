function Compute_LID(filename, instances_in_use, k_Estimator, dataset_name, save_path)

%parameters
size_of_point = 8;
color_of_point = 'r';

feature = fvecs_read(filename);
feature = transpose(feature);

[instances, length] = size(feature);
index = randperm(instances, instances_in_use);

count = 0;
LID_MLE = zeros([1, instances_in_use]);
LID_RV = zeros([1, instances_in_use]);

time_start = cputime;
%if the distance is recomputed by the RC program, then we can direcrtly use
%it.
for feature_index = index
   IDx = 0;
   count = count +1;
   %select the 'query' vector
   vector = feature(feature_index, :);
   %compute the distance
   Distance = sqrt(vector.^2*ones(size(feature'))+ones(size(vector))*(feature').^2-2*vector*feature');
   Distance = sort(Distance);
   Distance = Distance(2:end);
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
   
   if mod(count, 100) == 0
       disp(count);
       time_check = cputime-time_start;
       disp(time_check);
   end
end

save([save_path, 'LID MLE ', num2str(k_Estimator), '.txt'], 'LID_MLE', '-ascii');
save([save_path, 'LID RV ',  num2str(k_Estimator), '.txt'], 'LID_RV', '-ascii');

%draw the figure with MLE
LID_mean = mean(LID_MLE);
LID_std = std(LID_MLE);
LID_median = median(LID_MLE);

figure
x = linspace(1, instances_in_use, instances_in_use);
scatter(x, LID_MLE, size_of_point, color_of_point, 'filled');

xticks([0, 2000, 4000, 6000, 8000, 10000]);
xticklabels({'0', '2K', '4K', '6K', '8K', '10K'});

xlabel('number of instances');
ylabel('LID MLE');
txt = {[' mean ', num2str(LID_mean), ' std ', num2str(LID_std)], [' median ', num2str(LID_median), ' k ', num2str(k_Estimator)]};
yl = ylim;
text(0, yl(2)-5, txt);
title(['the LID distribution of ', dataset_name, ' with MLE ']);
saveas(gcf, [save_path, 'LID MLE distribution ',  num2str(k_Estimator), '.png']);

%draw the figure with MLE
LID_mean = mean(LID_RV);
LID_std = std(LID_RV);
LID_median = median(LID_RV);

figure
x = linspace(1, instances_in_use, instances_in_use);
scatter(x, LID_RV, size_of_point, color_of_point, 'filled');

xticks([0, 2000, 4000, 6000, 8000, 10000]);
xticklabels({'0', '2K', '4K', '6K', '8K', '10K'});

xlabel('number of instances');
ylabel('LID RV');
txt = {[' mean ', num2str(LID_mean), ' std ', num2str(LID_std)], [' median ', num2str(LID_median), ' k ', num2str(k_Estimator)]};
yl = ylim;
text(0, yl(2)-5, txt);
title(['the LID distribution of ', dataset_name, ' with RV ']);
saveas(gcf, [save_path, 'LID RV distribution ',  num2str(k_Estimator), '.png']);














