function model = team_training_code(input_directory,output_directory, verbose) % train_EEG_classifier
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Purpose: Train EEG classifiers and obtain the models
% Inputs:
% 1. input_directory
% 2. output_directory
%
% Outputs:
% 1. model: trained model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if verbose>=1
    disp('Finding challenge data...')
end

% Find the folders
patient_ids=dir(input_directory);
patient_ids=patient_ids([patient_ids.isdir]==1);
patient_ids(1:2)=[]; % Remove "./" and "../" paths
patient_ids={patient_ids.name};
num_patients = length(patient_ids);

% Create a folder for the model if it doesn't exist
if ~isfolder(output_directory)
    mkdir(output_directory)
end
fprintf('Loading data for %d patients...\n', num_patients)

channels = {'Fp1', 'Fp2', 'F3','F4'};
test_chs_L=length(channels);
[PartA, PartB]=z_bipartition_TE(test_chs_L);

j_cnt=0;
for j=1:num_patients 

    if verbose>1
        fprintf('%d/%d \n',j,num_patients)
    end

    % Extract features
    patient_id=patient_ids{j};

    current_features=get_features(input_directory,patient_id,PartA,PartB,channels);

    if size(current_features,1)>0
        if sum(sum(isnan(current_features')))~=0
            current_features(sum(isnan(current_features'))>0,:)=[];
        end
        if size(current_features,1)>0
            c_d_num=[];
            for c_i=1:size(current_features,2)-1
                d_tmp=find(abs(current_features(:,c_i)) > 999);
                c_d_num=vertcat(c_d_num,d_tmp);
                clear d_tmp
            end
            if sum(c_d_num)~=0
                current_features(c_d_num,:)=[];
            end
            if size(current_features,1)>0

                if size(current_features,1)>0
                    j_cnt=j_cnt+1;

                    [patient_metadata,~]=load_challenge_data(input_directory,patient_id);
                    current_outcome=get_outcome(patient_metadata);

                    current_features(:,size(current_features,2)+1)=current_outcome;
                    features_struct{j_cnt}=current_features;

                end
            end
        end
    end    

    clear current_features patient_id hos_Num
    clear patient_metadata current_outcome c_d_num
end

%% train model
e_div_L=4;
T_D_L=3;
channel_L=4;

model_outcome=[];
features_all_72=[];
features_all_48=[];
features_all_24=[];

t_f_size=20;
for f_j=1:length(features_struct)
    current_features=features_struct{f_j};
    time_n=unique(current_features(:,end-1));
    s_current_features=[];
    for t_i=1:length(time_n)
        time_f_N=find(time_n(t_i)==current_features(:,end-1));
        if length(time_f_N) > t_f_size
            selected_f_N=time_f_N(randperm(length(time_f_N),t_f_size));
        else
            selected_f_N=time_f_N;
        end
        s_current_features=vertcat(s_current_features,current_features(selected_f_N,:));
        clear selected_f_N time_f_N
    end

    f24_tmp=s_current_features(s_current_features(:,end-1)<=24,:);
    f48_tmp=s_current_features(s_current_features(:,end-1)<=48 & s_current_features(:,end-1)>24,:);
    f72_tmp=s_current_features(s_current_features(:,end-1)<=72 & s_current_features(:,end-1)>48,:);

    features_all_24=[features_all_24;f24_tmp];
    features_all_48=[features_all_48;f48_tmp];
    features_all_72=[features_all_72;f72_tmp];

    clear current_features s_current_features
    clear f24_tmp f48_tmp f72_tmp full_tmp
end

features_all_24(:,end-1)=[];
features_all_48(:,end-1)=[];
features_all_72(:,end-1)=[];

% TE
fm_range_tmp{1}=1:T_D_L;
% power
fm_range_tmp{2}=T_D_L+1:T_D_L+channel_L;
% SE_div
fm_range_tmp{3}=T_D_L+channel_L+1:T_D_L+channel_L+e_div_L*2;
% SE_origin
fm_range_tmp{4}=T_D_L+channel_L+e_div_L*2+1:T_D_L+channel_L+e_div_L*4;
% SE power
fm_range_tmp{5}=T_D_L+channel_L+e_div_L*4+1:T_D_L+channel_L+e_div_L*6;

fm_range{1}=[fm_range_tmp{1} fm_range_tmp{3}];
fm_range{2}=[fm_range_tmp{1} fm_range_tmp{3} fm_range_tmp{4}];
fm_range{3}=[fm_range_tmp{1} fm_range_tmp{4} fm_range_tmp{5}];
fm_range{4}=1:size(features_all_72,2)-1;

f_m_size=length(fm_range);
time_m_size=3;

for m_i=1:time_m_size
    if m_i==1
        features_all=features_all_24;
    elseif m_i==2
        features_all=[features_all_24;features_all_48];
    elseif m_i==3
        features_all=[features_all_48;features_all_72];
    end

    if size(features_all,1) > 10
        for f_m_i=1:f_m_size
            model_outcome{m_i,f_m_i} = fitcsvm(features_all(:,fm_range{f_m_i}),features_all(:,end),'Standardize',true,'KernelFunction','RBF','KernelScale','auto');
        end
    end

    clear features_all
end
clear features_all_24 features_all_48 features_all_72


% train label check
results=[];

prob_threshold=0.5;
for j=1:length(features_struct)
    
    current_features=features_struct{j};
    current_outcome=current_features(1,end);

    % time feature ratio
    time_n=unique(current_features(:,end-1));
    s_current_features=[];
    for t_i=1:length(time_n)
        time_f_N=find(time_n(t_i)==current_features(:,end-1));
        selected_f_N=time_f_N;
        s_current_features=vertcat(s_current_features,current_features(selected_f_N,:));
        clear selected_f_N time_f_N s_num
    end

    t24_tmp=s_current_features(s_current_features(:,end-1)<=24,1:end-2);
    t48_tmp=s_current_features(s_current_features(:,end-1)<=48 & s_current_features(:,end-1)>24,1:end-2);
    t72_tmp=s_current_features(s_current_features(:,end-1)<=72 & s_current_features(:,end-1)>48,1:end-2);
    clear s_current_features

    outcome_p_all=ones(f_m_size,time_m_size)*-1;
    for m_i=1:time_m_size
        if m_i==1
            features=t24_tmp;
        elseif m_i==2
            features=[t24_tmp;t48_tmp];
        elseif m_i==3
            features=[t48_tmp;t72_tmp];
        end

        if sum(sum(features))~=0
            if size(features,1)>0
                decision_all=zeros(size(features,1),f_m_size);
                for mf_i=1:f_m_size
                    model_solo=model_outcome{m_i,mf_i};
                    decision_all(:,mf_i)= predict(model_solo,features(:,fm_range{mf_i}));
                    clear model_solo
                end
                % decision_all
                outcome_p_all(:,m_i)=mean(decision_all);
            end
            clear decision_all
        end
        clear features
    end
    % method별 weight
    outcome_w_all=outcome_p_all;
    outcome_w_all(1,:)=outcome_w_all(1,:)*0.24;
    outcome_w_all(2,:)=outcome_w_all(2,:)*0.25;
    outcome_w_all(3,:)=outcome_w_all(3,:)*0.25;
    outcome_w_all(4,:)=outcome_w_all(4,:)*0.26;
    outcome_all=sum(outcome_w_all);

    % 시간대별 weight
    avail_m=find(outcome_all(1,:) >= 0);
    if length(avail_m)==3
        w_p=[0.3 0.4 0.3];
    elseif length(avail_m)==2
        if length(intersect(avail_m,[1 2]))==2
            w_p=[0.4 0.6 0];
        elseif length(intersect(avail_m,[1 3]))==2
            w_p=[0.5 0 0.5];
        elseif length(intersect(avail_m,[2 3]))==2
            w_p=[0 0.6 0.4];
        end
    elseif length(avail_m)==1
        if avail_m==1
            w_p=[1 0 0];
        elseif avail_m==2
            w_p=[0 1 0];
        elseif avail_m==3
            w_p=[0 0 1];
        end
    end
    out_probability=outcome_all*w_p';

    % current_outcome
    if out_probability <= prob_threshold
        outcome_binary=0;
    else
        outcome_binary=1;
    end

    results(j,1)=current_outcome;
    results(j,2)=outcome_binary;

    clear avail_m w_p time_n
    clear current_features patient_id features current_outcome outcome_all
    clear outcome_probability patient_metadata outcome_binary outcome_p_all
    clear t48_tmp t24_tmp t72_tmp outcome_binary_all out_probability outcome_w_all

end



% sub modeling

% sub model #1 good
sub_good_j=find(results(:,1)==0);
good_error_N=find(results(:,1)-results(:,2)==1);
sub_model1_N=[sub_good_j; good_error_N];
features_all_72=[];
features_all_48=[];
features_all_24=[];

for f_j=1:length(sub_model1_N)
    current_features=features_struct{sub_model1_N(f_j)};
    time_n=unique(current_features(:,end-1));
    % time feature ratio
    s_current_features=[];
    for t_i=1:length(time_n)
        time_f_N=find(time_n(t_i)==current_features(:,end-1));
        if length(time_f_N) > t_f_size
            selected_f_N=time_f_N(randperm(length(time_f_N),t_f_size));
        else
            selected_f_N=time_f_N;
        end
        s_current_features=vertcat(s_current_features,current_features(time_f_N,:));
        clear selected_f_N time_f_N s_num
    end
    f24_tmp=s_current_features(s_current_features(:,end-1)<=24,:);
    f48_tmp=s_current_features(s_current_features(:,end-1)<=48 & s_current_features(:,end-1)>24,:);
    f72_tmp=s_current_features(s_current_features(:,end-1)<=72 & s_current_features(:,end-1)>48,:);
    features_all_24=[features_all_24;f24_tmp];
    features_all_48=[features_all_48;f48_tmp];
    features_all_72=[features_all_72;f72_tmp];

    clear current_features s_current_features
    clear f24_tmp f48_tmp f72_tmp
end
features_all_24(:,end-1)=[];
features_all_48(:,end-1)=[];
features_all_72(:,end-1)=[];

sub_good_model=[];
for m_i=1:time_m_size
    % m_i
    if m_i==1
        features_all=features_all_24;
    elseif m_i==2
        features_all=[features_all_24;features_all_48];
    elseif m_i==3
        features_all=[features_all_48;features_all_72];
    end

    if size(features_all,1) > 10
        for f_m_i=1:f_m_size
            % f_m_i
            sub_good_model{m_i,f_m_i} = fitcsvm(features_all(:,fm_range{f_m_i}),features_all(:,end),'Standardize',true,'KernelFunction','RBF','KernelScale','auto');
            clear f_m_i
        end
    end
    clear features_all m_i
end
clear features_all_24 features_all_48 features_all_72 sub_model1_N
clear sub_good_j good_error_N


% sub model #2 bad
sub_poor_j=find(results(:,1)==1);
poor_error_N=find(results(:,1)-results(:,2)==-1);
sub_model2_N=[sub_poor_j; poor_error_N];
features_all_72=[];
features_all_48=[];
features_all_24=[];
for f_j=1:length(sub_model2_N)
    current_features=features_struct{sub_model2_N(f_j)};
    time_n=unique(current_features(:,end-1));
    % time feature ratio
    s_current_features=[];
    for t_i=1:length(time_n)
        time_f_N=find(time_n(t_i)==current_features(:,end-1));
        if length(time_f_N) > t_f_size
            selected_f_N=time_f_N(randperm(length(time_f_N),t_f_size));
        else
            selected_f_N=time_f_N;
        end
        s_current_features=vertcat(s_current_features,current_features(time_f_N,:));
        clear selected_f_N time_f_N s_num
    end
    f24_tmp=s_current_features(s_current_features(:,end-1)<=24,:);
    f48_tmp=s_current_features(s_current_features(:,end-1)<=48 & s_current_features(:,end-1)>24,:);
    f72_tmp=s_current_features(s_current_features(:,end-1)<=72 & s_current_features(:,end-1)>48,:);
    features_all_24=[features_all_24;f24_tmp];
    features_all_48=[features_all_48;f48_tmp];
    features_all_72=[features_all_72;f72_tmp];

    clear current_features s_current_features
    clear f24_tmp f48_tmp f72_tmp time_n
end
features_all_24(:,end-1)=[];
features_all_48(:,end-1)=[];
features_all_72(:,end-1)=[];

sub_poor_model=[];
for m_i=1:time_m_size
    % m_i
    if m_i==1
        features_all=features_all_24;
    elseif m_i==2
        features_all=[features_all_24;features_all_48];
    elseif m_i==3
        features_all=[features_all_48;features_all_72];
    end

    if size(features_all,1) > 10
        for f_m_i=1:f_m_size
            % f_m_i
            sub_poor_model{m_i,f_m_i} = fitcsvm(features_all(:,fm_range{f_m_i}),features_all(:,end),'Standardize',true,'KernelFunction','RBF','KernelScale','auto');
            clear f_m_i
        end
    end
    clear features_all m_i
end
clear features_all_24 features_all_48 features_all_72 sub_model2_N
clear sub_poor_j poor_error_N s_num




% model save
model_cpc=model_outcome;
% save_model(model_outcome,model_cpc,output_directory);
save_model(model_outcome,sub_good_model,sub_poor_model,model_cpc,output_directory);

end

%% functions
%---------------------------------------------
function save_model(model_outcome,sub_good_model,sub_poor_model,model_cpc,output_directory)
% Save results.
filename = fullfile(output_directory,'model.mat');
save(filename,'model_outcome','sub_good_model','sub_poor_model','model_cpc','-v7.3');
disp('Done.')
end

%---------------------------------------------
function outcome=get_outcome(patient_metadata)
patient_metadata=strsplit(patient_metadata,'\n');
outcome_tmp=patient_metadata(startsWith(patient_metadata,'Outcome:'));
outcome_tmp=strsplit(outcome_tmp{1},':');
if strncmp(strtrim(outcome_tmp{2}),'Good',4)
    outcome=0;
elseif strncmp(strtrim(outcome_tmp{2}),'Poor',4)
    outcome=1;
else
    keyboard
end
end

%---------------------------------------------
function [PartA, PartB]=z_bipartition_TE(channel_length)
G = 1:channel_length;
f = 1;
for N = 2:length(G)
    cases = nchoosek(G, N);
    maxM = N-1; %
    for idxD=1:size(cases,1)
        m = 0; %
        for idxC=1:maxM
            tmp = nchoosek(1:N, idxC);
            for idxS=1:size(tmp,1)

                m = m + 1;
                PartA{f,1}{idxD,m} = cases(idxD,tmp(idxS,:));
                PartB{f,1}{idxD,m} = setdiff(cases(idxD,:),PartA{f,1}{idxD,m});
            end
        end
    end
    f = f+1;
end
end


