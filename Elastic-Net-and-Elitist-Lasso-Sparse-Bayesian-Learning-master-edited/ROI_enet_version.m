% clc
% clear all
% close all

%% This script is for elastic net predction using sparse bayesian learning

Face_RT=behavdata1((1:336),1);%response variable (Y)
Face_RT=table2array(Face_RT);

%% divide into train and test set
idx = randperm(length(Face_RT));
Face_train_num=idx(1:round(0.8*length(Face_RT)));
Face_test_num=idx(round(0.8*length(Face_RT))+1:end);
    
%% divide train set and test set   
    HCP_con_train=Avg_Fear_Fx_N(Face_train_num,:); %predictors (x)
    HCP_RT_train=Face_RT(Face_train_num,:);
   
    HCP_con_test=Avg_Fear_Fx_N(Face_test_num,:);
    HCP_RT_test=Face_RT(Face_test_num,:);
    
%% 
no_sub=size(HCP_con_train,2);
miu=zeros(no_sub,1);
alpha1=zeros(no_sub,1);
alpha2=zeros(no_sub,1);
 %% applying enet ssbl
 
[miu,alpha1,alpha2]=enet_ssbl(HCP_RT_train,HCP_con_train);

%% fit the model on the test set
LM = fitlm(miu,
model = fitlm(miu, HCP_RT_train);
 
 
