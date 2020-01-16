
function draw_figure(vector, name_of_vector, name_of_dataset, regression, max, min, mean, dimension)

length = size(vector);

x = zeros(length);
y = zeros(length);
y_gradient = zeros(length);

length = length(2);
index_num = 0;

for i = 1:length
    x(i) = vector(i);
    index_num = index_num + 1;
    y(i) = index_num;
end

y = y / length;
for i = 1:length
    if y(i) < 0
        y(i) = 0;
    end
end
yy = smooth(x, y, regression, 'loess');


for j = 1: length-1
    y_gradient(j) = (yy(j+1)-yy(j))/(x(j+1)-x(j));
end
y_gradient(length) = 0;

yy_gradient = smooth(x, y_gradient, regression, "loess");

figure 
yyaxis left;
plot(x, yy);
yyaxis right;
plot(x, yy_gradient, "--");
yl = ylim;
xl = xlim;
grid on

if nargin>=5
    txt = {['max ',num2str(max),' min ',num2str(min)], ['mean ',num2str(mean), ' dimension ', num2str(dimension)]};
    text(xl(1), yl(2)-2, txt)
end

legend("Cumulative Distribution", "Probability Density", 'Location', 'northwest')
title([name_of_vector, "in", name_of_dataset])



