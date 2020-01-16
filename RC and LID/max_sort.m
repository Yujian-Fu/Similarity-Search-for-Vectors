
function result = max_sort(vector)

length = size(vector);
length = length(2)
for i = 1:length
    max = 0;
    index = 0;
    for j = i:length
        if vector(j) > max
            max = vector(j);
            index = j;
        end
    end
    temp = vector(i);
    vector(i) = max;
    vector(index) = temp;
end

result = vector;
