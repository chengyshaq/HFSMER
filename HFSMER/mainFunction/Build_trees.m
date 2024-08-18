function [trees]= Build_trees (count_num,K,data_array,stra)
count=count_num(:,2);
a=length(count);
tree=[];
allnum=sum(count);
epss=1;
num_clusters=K;
c=0;
[label_clusters,~] = cluster(data_array, 1, num_clusters,stra);
for i=1:K
    node_dex=[];
    node_dex=find(label_clusters(:,2)==i);
    if length(node_dex)>1;
        c=c+1;
        count(real(a+c),1)=sum(count(node_dex,1));
        tree(node_dex,1)=real(a+c);
        count(node_dex,1)=inf;
    end
end
for j=1:10000
    if epss<allnum-1
        [~, sorted_indices] = sort(count);
        epss=sum(count(sorted_indices(1:K,1),1));
        c=c+1;
        count(real(a+j),1)=sum(count(sorted_indices(1:K,1),1));
        tree(sorted_indices(1:K,1),1)=real(a+c);
        count(sorted_indices(1:K,1),1)=inf;
    else
        break;
    end
    
end
b=length(tree);
tree(b+1,1)=0;
tree(b+1,2)=0;
leval=0;
nodes=b+1;
D = [];
while (nodes~=0)
    leval=leval+1;
    nnodes = [];
    for k = 1 : length(nodes)
        nnodes = [nnodes; find(tree(:,1)==nodes(k))];
    end
    D = [D; nnodes];
    nodes = nnodes; 
    tree(nnodes,2)=leval;
end
c=length(D);
trees=tree;

end