function [labels,data_array] = cluster(data_array, sigma, num_clusters,stra)
features = data_array(:, 1:end-1);
unique_labels = unique(data_array(:, end));

label_to_indices = containers.Map('KeyType', 'double', 'ValueType', 'any');
for i = 1:length(unique_labels)
    label_to_indices(unique_labels(i)) = find(data_array(:, end) == unique_labels(i));
end

label_means = zeros(length(unique_labels), size(features, 2));
for i = 1:length(unique_labels)
    label_data = features(label_to_indices(unique_labels(i)), :);
    [~,label_means(i, :)] = kmeans(label_data, 1);
    label_means(i, :) = mean(label_data, 1);
end
granularity_weights = zeros(length(unique_labels), 1);
for i = 1:length(unique_labels)
    granularity_weights(i) = 1 / length(label_to_indices(unique_labels(i)));
end
S = zeros(length(unique_labels));
for i = 1:length(unique_labels)
    for j = 1:length(unique_labels)
        if i ~= j
            original_similarity = exp(-norm(label_means(i,:) - label_means(j,:))^2 )/ (2*sigma^2);
            granularity_similarity= sqrt(granularity_weights(i)*granularity_weights(j));
            if stra==1
                S(i,j)=power((2*original_similarity*granularity_similarity )/(original_similarity+granularity_similarity),exp(-1));                      
            else
                S(i,j)=original_similarity*granularity_similarity;
            end
            [m, n] = size(S);
        end
    end
end
D = diag(sum(S, 2));
L = D - S;
if (isequal(L, L'))    
    [V, ~] = eigs(L, num_clusters, 'SA');   
else
    [V, ~] = eigs(L, num_clusters, 'SM');
end
label_clusters = kmeans(V, num_clusters);

unique_labels(:,2)=label_clusters ;
labels=unique_labels;
label_to_cluster = containers.Map(unique_labels(:,1), unique_labels(:,2));
for i = 1:size(data_array, 1)
    current_label = data_array(i, end);
    cluster_number = label_to_cluster(current_label);
    data_array(i, end) = cluster_number;
end
end
