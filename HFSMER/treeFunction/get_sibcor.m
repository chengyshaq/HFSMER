function [sibcor,Lcor] = get_sibcor(X,hiercor,Y,tree)
%������������������ƶȼ����ֵܽڵ������ƶ�,�����ֵ���0��ʾ
sigma=1;
internalNodes = tree_InternalNodes(tree);
indexRoot = tree_Root(tree);% The root of the tree
noLeafNode =[internalNodes;indexRoot];
internalNodes = tree_InternalNodes(tree);
internalNodes(find(internalNodes==-1))=[];
sibcor=zeros(length(Y),length(Y));
Lcor=[];
unique_labels=1:length(X);
for i = 1:length(unique_labels)% ����ÿ����ǩ��������ֵ
    label_means(i, :) = mean(X{i}, 1);
end
for i = 1:length(unique_labels)
    for j = 1:length(unique_labels)
        if i ~= j
            % ����ԭʼ�ĸ�˹�����ƶ�
            sibcor(i,j) = exp(-norm(label_means(i,:) - label_means(j,:))^2 / (2*sigma^2));
        end
    end
end
for i = 1:length(noLeafNode)
    m(noLeafNode(i)) = length(find(tree(:,1)==noLeafNode(i)));
end
maxm=max(m);
% for i = 1:length(noLeafNode)
%     children_set = get_children_set(tree, noLeafNode(i));
%     if ~isempty(children_set)
%         S = zeros(length(unique_labels));
%         for i = 1:length(unique_labels)
%             for j = 1:length(unique_labels)
%                 if i ~= j
%                     % ����ԭʼ�ĸ�˹�����ƶ�
%                     original_similarity = exp(-norm(label_means(i,:) - label_means(j,:))^2 / (2*sigma^2));
%                     % Ӧ������Ȩ��
%                 end
%             end
%         end
% %         YY{noLeafNode(i)}=myconversionY01(Y{noLeafNode(i)},length(children_set),children_set);%extend 2 to [1 0]
%     end
% end
for k=1:length(noLeafNode)
    if isempty(Y{noLeafNode(k)})
        continue
    end
    cur_child = get_children_set(tree,noLeafNode(k));
    if isempty(cur_child)
        continue
    end
    LLcor=[];
     for i = 1:length(cur_child)
            for j = 1:length(cur_child)
                if i ~= j
                    % ����ԭʼ�ĸ�˹�����ƶ�
                    LLcor(i,j)= sibcor(cur_child(i),cur_child(j));  %�Խ�ȡ0
                end
            end
     end
     D=[];
     D = diag(sum(LLcor, 2));
     Lcor{noLeafNode(k)}=D-LLcor;
% ����������˹����
%     cur_par=tree(noLeafNode(i),1);
%     cur_index=find(Y{cur_par}==internalNodes(i));
%     cur_cor=hiercor{cur_par}(cur_index,:);
%     for j=1:length(cur_sib)
%         sib_index=find(Y{cur_par}==cur_sib(j));
%         if isempty(sib_index)
%             continue
%         end
%         cur_sib_cor=cur_cor(:,sib_index);
%         sibcor(internalNodes(i),cur_sib(j))=mean(mean(cur_sib_cor,2));
%     end
end
% temp=sibcor;
% temp_index=find(temp==0);
% temp(temp_index)=[];
% if length(temp)>0
%     sibcor=(sibcor-min(temp))./(max(temp)-min(temp));
% end
% sibcor(temp_index)=0;
end

