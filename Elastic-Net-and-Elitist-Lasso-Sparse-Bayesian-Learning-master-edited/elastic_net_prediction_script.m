% clc
% clear all
% close all

%% This script is for elastic net predction using sparse bayesian learning
%% response variable (Y)
Face_RT=behavdata1((1:336),1);
Face_RT=table2array(Face_RT);

%% divide into train and test set
idx = randperm(length(Face_RT));
Face_train_num=idx(1:round(0.8*length(Face_RT)));
Face_test_num=idx(round(0.8*length(Face_RT))+1:end);
    
%% divide train set and test set   
    HCP_con_train=Em_F(:,:,Face_train_num);
    HCP_RT_train=Face_RT(Face_train_num,:);
   
    HCP_con_test=Em_F(:,:,Face_test_num);
    HCP_RT_test=Face_RT(Face_test_num,:);
    
%% For the train set and the test set, take only the upper triangular part of the connectivity
nedgesFC=((size(HCP_con_train,1)*size(HCP_con_train,1))-size(HCP_con_train,1))/2;
        %young
        train_FCrs0=zeros(nedgesFC,size(HCP_con_train,3));
                            for a =1:size(HCP_con_train,3)
                                CM=HCP_con_train(:,:,a);
                                lower_triFCmask = tril(true(size(CM)),-1);
                                train_FCrs0(:,a)=CM(lower_triFCmask);
                            end
        row0FC=find(sum(train_FCrs0,2)==0);
        train_FCrs0(row0FC,:)=[]; 
        
 HCP_con_trainF=train_FCrs0;
 
 %% test set (lower triangular of conn)
 nedgesFC=((size(HCP_con_test,1)*size(HCP_con_test,1))-size(HCP_con_test,1))/2;
        %young
        test_FCrs0=zeros(nedgesFC,size(HCP_con_test,3));
                            for a =1:size(HCP_con_test,3)
                                CM=HCP_con_test(:,:,a);
                                lower_triFCmask = tril(true(size(CM)),-1);
                                test_FCrs0(:,a)=CM(lower_triFCmask);
                            end
        row0FC=find(sum(test_FCrs0,2)==0);
        test_FCrs0(row0FC,:)=[]; 
        
 HCP_con_testF=test_FCrs0;
 
 %% applying enet ssbl
[miu,alpha1,alpha2]=enet_ssbl(HCP_RT_train,HCP_con_trainF);
 
 
 
