function [feature_slct,W] = HFSMER(X, Y, tree, Lcor, treecor,sibcor,lambda1, lambda2, lambda3, lambda4,gamma,flag)
internalNodes = tree_InternalNodes(tree);
indexRoot = tree_Root(tree);% The root of the tree
noLeafNode =[internalNodes;indexRoot];
[r,c]=size(treecor);
[r2,c2]=size(sibcor);
v1=ones(r,c);
v2=ones(r2,c2);
treecor=v1-treecor;
sibcor=v2-sibcor;
optmParameter.lambda1    = lambda1;    
optmParameter.lambda2    = lambda2;                                                                                                                                                                                                                                                                                          ; % 2.^[-10:10] % sparsity
optmParameter.lambda3    = lambda3;  
optmParameter.lambda4    = lambda4;   
optmParameter.gamma      = gamma;       %  Initialize W
optmParameter.maxIter      =100; 
optmParameter.minimumLossMargin  = 10^-6;
optmParameter.bQuiet             = 1;
for i = 1:length(noLeafNode)
    children_set = get_children_set(tree, noLeafNode(i));
    m(noLeafNode(i)) = length(children_set);
end
maxm=max(m);
for i = 1:length(noLeafNode)
    children_set = get_children_set(tree, noLeafNode(i));
    m(noLeafNode(i)) = length(children_set);
    Yy{noLeafNode(i)}=myconversionY01(Y{noLeafNode(i)},length(children_set),children_set);%extend 2 to [1 0]
    XXX{i}=X{noLeafNode(i)};
    cur_childern=get_children_set(tree,noLeafNode(i));
    YYY{i}= myconversionY01(Y{noLeafNode(i)}, length(cur_childern),cur_childern);
    LL{i}=Lcor{noLeafNode(i)};
    W{noLeafNode(i)}=LSGS( XXX{i},YYY{i},LL{i}, optmParameter,maxm);
end
for i = 1: length(noLeafNode)
    W1=W{noLeafNode(i)};
    W{noLeafNode(i)} = W1(:,1:m(noLeafNode(i)));
end

for j = 1: length(noLeafNode)
    tempVector = sum(W{noLeafNode(j)}.^2, 2);
    [atemp, value] = sort(tempVector, 'descend'); % sort tempVecror (W) in a descend order
    clear tempVector;
    feature_slct{noLeafNode(j)} = value(1:end);
end
if (flag == 1)
    fontsize = 20;
    figure1 = figure('Color',[1 1 1]);
    axes1 = axes('Parent',figure1,'FontSize',fontsize,'FontName','Times New Roman');
    
    plot(obj,'LineWidth',4,'Color',[0 0 1]);
    xlim(axes1,[0.8 10]);
    %     ylim(axes1,[16000,36000]);%Cifar
    % set(gca,'yscale','log')
    set(gca,'FontName','Times New Roman','FontSize',fontsize);
    xlabel('Iteration number');
    ylabel('Objective function value');
end
end



