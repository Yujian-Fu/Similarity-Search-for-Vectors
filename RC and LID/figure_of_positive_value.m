

function positive_probability = figure_of_positive_value(v, name_of_dataset, regression1, regression2)

[length , instances] = size(v);

vector_state = zeros(1, instances);
element_state = zeros(1, length);

max_value = max(v(:));
min_value = min(v(:));
mean_value = mean(v(:));
mode_value = mode(v(:));
dimension = length;
number = 0;
for i = 1:length
    for j = 1:instances
        if v(i, j) == 0
            vector_state(j) = vector_state(j) +1;
            element_state(i) = element_state(i) +1;
            number = number + 1;
        end
    end
end

vector_state = sort(1 - vector_state / length);
element_state = sort(1 - element_state / instances);

draw_figure(vector_state, "distribution of vectors", name_of_dataset, regression1, max_value, min_value, mean_value, dimension);
draw_figure(element_state, "distribution of elements", name_of_dataset, regression2);
    
    