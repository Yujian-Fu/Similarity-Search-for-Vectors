

function rc = Relative_Contrast(file_name, instances_in_use, dataset_name, save_path)

feature = fvecs_read(file_name);
feature = transpose(feature);

[instances, length] = size(feature);

index = randperm(instances, instances_in_use);

rc = zeros([1, instances_in_use]);

time_start = cputime;
count = 0;
%to save time and allow LID can use such distance result, we save the
%computation result.
distance_result = zeros([instances_in_use, 5000]);
for i = index
   count = count + 1;
   vector = feature(i, :);
   Distance = sqrt(vector.^2*ones(size(feature'))+ones(size(vector))*(feature').^2-2*vector*feature');
   Distance = sort(Distance);
   Distance = Distance(2:end);
   distance_result(count, :) = Distance(1:5000);
   j = 1;
   while (Distance(j) == 0)
       j = j+1;
   end
   Distance = Distance(j:end);
   Dmean = mean(Distance);
   Dmin = Distance(1);
   rc(count) = Dmean/Dmin;
   save([save_path, 'distance_matrix.mat'], 'distance_result', '-mat');
   save([save_path, 'RC vector.mat'], 'rc', '-mat');
   if mod(count, 5) == 0
       disp(count);
       time_check = cputime - time_start;
       disp(time_check);
   end
end
save([save_path, 'distance_matrix.mat'], 'distance_result', '-mat');
save([save_path, 'RC vector.mat'], 'rc', '-mat');
rc_mean = mean(rc);
rc_std = std(rc);
rc_median = median(rc);

figure
x = linspace(1, instances_in_use, instances_in_use);
scatter(x, rc, 8, 'r', 'filled');
xticks([0, 400, 800, 1200, 1600, 2000]);
%xticklabels({'0', '2K', '4K', '6K', '8K', '10K'});

xlabel('number of instances');
ylabel('RC value');
txt = {[' mean ', num2str(rc_mean), ' std ', num2str(rc_std)], [' median ', num2str(rc_median)]};
yl = ylim;
text(0, yl(2)*0.8, txt);
title(['the RC distribution of ', dataset_name]);

saveas(gcf, [save_path, 'RC distribution.png']);



