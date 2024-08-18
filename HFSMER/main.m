clc; clear;
 str={'DD';'F194';'CLEF';'VOC';'ISL57';'SUN';'Cifar37'};
str1={str{1}};
%% Initialisation parameters 
tic;
stra    = 1;  %  stra=1  MER strategy / stra=0 MGE strategy
K       = 3;  %  superclass number
lambda1 = 10; %L21_norm
lambda2 = 0.1; %L1_norm
lambda3 = 1; %global correlation
lambda4 = 0.1; %local correlation
gamma   = 100; % Initialisation W
m = length(str1);
rng('default');
indx=[];
filename = [str1{1} 'Train']
load (filename);
count_num=tabulate(data_array(:,end));
new_trees=Build_trees(count_num,K,data_array,stra);
 tree=new_trees;
cor=corr(data_array(:,1:end-1)',data_array(:,1:end-1)','type','pearson');
[X,Y,~,cor]=create_SubTable(data_array, tree,cor);
unique_labels = unique(data_array(:, end));
label_to_indices = containers.Map('KeyType', 'double', 'ValueType', 'any');
for o = 1:length(unique_labels)
    label_to_indices(unique_labels(o)) = find(data_array(:, end) == unique_labels(o));
    X{unique_labels(o)}=data_array(label_to_indices(unique_labels(o)),1:end-1);
end
TDTime= toc;
%% train
tic;
[treecor] = get_treecor(tree);
[sibcor,Lcor] = get_sibcor(X,cor,Y,tree);
[feature_slct,W] = HFSMER(X, Y, tree, Lcor, treecor,sibcor,lambda1, lambda2, lambda3, lambda4,gamma,0);
%% test
FSTime =toc;
count_num=tabulate(data_array(:,end));
max_num=max(count_num(:,2));
tail_num=max_num*0.2;
tail_index=find(count_num(:,2)<=tail_num);
testFile = [str1{1}, 'Test.mat']
load (testFile);
 tree=new_trees;
[accuracyMean, accuracyStd, F_LCAMean, FHMean, TIEmean, TestTime,accuracy_l,accuracy_tail,FHStd, TIEStd,accuracy_tailStd,FH,TIE] = HierSVMPredictionBatch(data_array, tree, feature_slct,tail_index,str1{1});        %
[t_r,~]=size(data_array);
tiemean=TIEmean/t_r;
tieStd=TIEStd./t_r;
tie=TIE./t_r;
Timeall= TestTime+TDTime +FSTime;
