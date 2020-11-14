%% Exercise: RSA
%
%  Instructions
%  ------------
% 
%  This file contains code that covers RSA.
%
%   
%   To run the code, you can copy each chunk and paste it in the Matlab
%   terminal, or highlight the specific lines and evaluate code.
%
%   Data used in this exercise are taken from Connolly et al 2012,
%   downloaded from CosmoMVPA's github.
%
%   Edited by JH POH - 2/16/2020
%
%   ********* Important Notes **********
%   Please note that many of the code and function implemented here have
%   been simplified to make things readable and understandable. You may
%   notice that there several hard-coded parameters and these
%   are mostly done for simplicity (and kinda for my own convenience).
%   Some of the code are directly taken from different classes I have taken
%   over the years and have not been extensively tested, so DO NOT
%   use these for your own actual data analysis!!!
%
%   Functions described here are all common functions that have been built
%   into various well-established packages. Do use those instead! Once you
%   understand the parameters required for each of these functions, it's
%   super easy to just call those functions regardless of package.
%
%
%% Initialization
clear ; close all; clc

%% ------------------- Part 1 (RSA - Models) ---------------------
%% Load Data 1
% The data is a set of t-maps from a single subject taken from Connolly et
% al (2012). This is part of the dataset used in the tutorials for
% CosmoMVPA.
% 
% In this dataset, participants were shown images from 6 categories
% (monkey, lemur, mallard, warbler, ladybug, lunamoth) across 10 runs.
% There are 2 species from 3 biological classes: Primates, Birds &
% Insects.
%
% The t-maps for each category (in each run) has been computed and compiled
% as a single 4D- nii file. To speed things up, I have already extracted
% the nii file into a .mat file. If you like, you can read in the .nii
% yourself and extract the data using functions from SPM like spm_vol or
% MRIread from Freesurfer. I personally prefer MRIread as it tends to be
% much faster. Do take note to check the orientation of the images,
% as different functions can store the dimensions differently.
%
% We also include a behavioral data matrix which indicates rating of
% pairwise dissimilarity across categories (human rating).
%
% For this exercise, we will only be looking at the Ventral visual
% cortices and also the early visual cortex.
%

% Load the tmaps
fmripath = [pwd '/fMRI_data_Connolly2012/'];
load([fmripath 'glm_T_stats_perrun.mat']);

% Load the mask for ventral visual cortex
load([fmripath 'vt_mask.mat']);

% Load the mask for early visual cortex
load([fmripath 'ev_mask.mat']);

%% Feature extraction
% In this demonstration, we are looking at a specific ROI, the ventral
% visual cortex (VT). So we first extract voxels that are in the ventral
% visual cortex by using a binary VT mask.
% We then organise these values into a single matrix where each row
% corresponds to 1 t-map, and each column corresponds to a feature (voxel).

% Calculating the number of voxels in the VT mask and initialise an empty
% matrix to store the activation pattern for each condition. 
% There are a total of 10 runs, so we average across the 10 runs for each
% condition, generating a 6 * M matrix (with M being the number of
% voxels/features).

% Extract the data for the VT mask.
numvx = length(find(vt_m==1));
vtmat = NaN(6,numvx);

for c = 1 : 6
    ct = 1;
    tmp = NaN(10,numvx);
    for i = c : 6 : 60
        tm = t(:,:,:,i);
        tmp(ct,:) = tm(logical(vt_m));
        ct = ct + 1;
    end
    vtmat(c,:) = nanmean(tmp,1);
end

% Extract the data for the EV mask.
numvx = length(find(ev_m==1));
evmat = NaN(6,numvx);

for c = 1 : 6
    ct = 1;
    tmp = NaN(10,numvx);
    for i = c : 6 : 60
        tm = t(:,:,:,i);
        tmp(ct,:) = tm(logical(ev_m));
        ct = ct + 1;
    end
    evmat(c,:) = nanmean(tmp,1);
end


%% Generating a RDM
% We have now generated a matrix, where each row corresponds to the
% activation pattern for each item. The row indices corresponds to the
% following conditions:
% Rows 1 : Monkey
% Rows 2 : Lemur
% Rows 3 : Mallard
% Rows 4 : Warbler
% Rows 5 : Ladybug
% Rows 6 : Lunamoth

% Create the RDM for the VT roi using correlation distance (1 - r) and
% visualise the RDM 
vt_rdm = squareform(pdist(vtmat,'correlation'));
figure;imagesc(vt_rdm);
title('VT RDM')
labels={'monkey','lemur','mallard','warbler','ladybug','lunamoth'}';
set(gca,'XTick',1:6,'XTickLabel',labels)
set(gca,'YTick',1:6,'YTickLabel',labels)

% In addition to the RDM, we can also visualise the data using a
% dendrogram, where the different categories are hierarhically separated
% based on the similarity in their representations. 
% The function 'linkage' creates a hierarchical clustering and this is then
% plotted using 'dendrogram'.
vt_links = linkage(vt_rdm,'single');
figure;dendrogram(vt_links,'labels',labels)

% Note: Another popular way of visualising RDM is by using multidimensional
% scaling (MDS). MDS can transform data from N-dimensional space into 2
% dimnesions, while preserving the relationship between all items. You can
% try getting the MDS using the function 'mdscale' on your RDM.


% ****** EXERCISE TIME! ********
% In the above analysis, we used a correlation distance. Try using
% different distance measures to see how the results may differ!
% (To look at the difference distance measures that the "pdist" function
% takes, type "help pdist" in the command window. If you like, pdist can 
% also take in any distance measure you define.)
%
% We created the RDM for the Ventral visual cortex ROI. Try creating the
% RDM for the Early visual cortex and visualise the results. Do the results
% look different? 
%
% ------------ Your code here -----------
%
ev_rdm = squareform(pdist(evmat,'correlation'));
figure;imagesc(ev_rdm);
title('EV RDM')
labels={'monkey','lemur','mallard','warbler','ladybug','lunamoth'}';
set(gca,'XTick',1:6,'XTickLabel',labels)
set(gca,'YTick',1:6,'YTickLabel',labels)








%% Model Comparison
% One of the most powerful aspect of RSA, is the ability to compare your
% RDM with different models or relational matrix.
% 
% In this exercise, we have 2 separate models from different modalities.
% The 1st model (behav_sim.mat), consist of similarity rating by human
% participants, while the 2nd model (v1_model.mat) is a decomposition of
% the stimulus image using spatial filters that mimic V1 neurons.

% Loading the models
load([fmripath 'behav_sim.mat']);
load([fmripath 'v1_model.mat']);

% Visualising the models.
figure;imagesc(behav);
title('Behavioral Model');
labels={'monkey','lemur','mallard','warbler','ladybug','lunamoth'}';
set(gca,'XTick',1:6,'XTickLabel',labels)
set(gca,'YTick',1:6,'YTickLabel',labels)

figure;imagesc(v1_model);
title('V1 Model');
labels={'monkey','lemur','mallard','warbler','ladybug','lunamoth'}';
set(gca,'XTick',1:6,'XTickLabel',labels)
set(gca,'YTick',1:6,'YTickLabel',labels)

% Comparing behavioral Model with fMRI RDMs using rank-ordered
% correlations.
% Since the matrices are symmetrical, we will take only the lower half (or
% upper if you prefer) of each matrix.
idx = logical(tril(ones(6,6),-1));
vt_vect = vt_rdm(idx);
behav_vect = behav(idx);
v1_vect = v1_model(idx);

% Over here, we are using Kendall's tau to correlate the different RDMs.
% Spearman is also commonly used, but Kendall's is typically preferred when
% correlating smaller datasets.
%
% There are also other methods that you can use, which certainly includes
% looking at parameter fits and/or model comparisons.
%
[rho, pval] = corr(vt_vect,behav_vect,'type','Kendall');
fprintf('Correlation between the VT RDM and the Behavioral model is : %f \n', rho);
fprintf('p-value: %f \n', pval)

[rho, pval] = corr(vt_vect,v1_vect,'type','Kendall');
fprintf('Correlation between the VT RDM and the V1 model is : %f \n', rho);
fprintf('p-value: %f \n', pval)



% ****** EXERCISE TIME! ********
% Now compare the 2 models with the Early visual cortex RDM, and see how it
% looks!
%
% 
% ------------ Your code here -----------
%
ev_vect = ev_rdm(idx);
[rho, pval] = corr(ev_vect,behav_vect,'type','Kendall');
fprintf('Correlation between the EV RDM and the Behavioral model is : %f \n', rho);
fprintf('p-value: %f \n', pval)

[rho, pval] = corr(ev_vect,v1_vect,'type','Kendall');
fprintf('Correlation between the EV RDM and the V1 model is : %f \n', rho);
fprintf('p-value: %f \n', pval)



% ****** MORE EXERCISE TIME! ********
% We have compared two different models looking at subjective rating of
% similarity (behav_sim), and also perceptual similarity based on image
% decomposition.
% Can you create a model based on the superordinate class of each species?
% (i.e. Insects, Birds and Primates)? 
% Create a model for the superordinate category and try matching that with
% the EVC and VT RDMs.
% 
% Hint: Remember that the matrices are computed using 'distances' and not
% similarity - i.e. smaller values = greater similarity.
%
% ------------ Your code here -----------
%
super_rdm = [
    0 0 1 1 1 1;
    0 0 1 1 1 1;
    1 1 0 0 1 1;
    1 1 0 0 1 1;
    1 1 1 1 0 0;
    1 1 1 1 0 0;
    ];
super_vect = super_rdm(idx);

[rho, pval] = corr(ev_vect,super_vect,'type','Kendall');
fprintf('Correlation between the EV RDM and the Superordinate model is : %f \n', rho);
fprintf('p-value: %f \n', pval)

[rho, pval] = corr(ev_vect,super_vect,'type','Kendall');
fprintf('Correlation between the EV RDM and the Superordinate model is : %f \n', rho);
fprintf('p-value: %f \n', pval)



% ****** EVEN MORE EXERCISE TIME!!! ********
% Can we create a model to perform a 'classification' type analysis?
% When performing classification analysis, we are trying to examine if 
% there is sufficient information that enables the discrimination of A and 
% B. 
% If our goal here is to examine if our ROI contains information that 
% allows the discrimination of vertebraes and invertebraes,
% how would we create this model?
%
% What about discriminating the different primates (Monkey vs Lemur)?
%(Hint: This may require computing a different RDM)
% ------------ Your code here -----------
%

% Vertebrae/Invertebrae comparison is similar to the above.
vert_rdm = [
    0 0 0 0 1 1;
    0 0 0 0 1 1;
    0 0 0 0 1 1;
    0 0 0 0 1 1;
    1 1 1 1 0 0;
    1 1 1 1 0 0;
    ];
vert_vect = vert_rdm(idx);

[rho, pval] = corr(vt_vect,vert_vect,'type','Kendall');
fprintf('Correlation between the VT RDM and the Vertebrae model is : %f \n', rho);
fprintf('p-value: %f \n', pval)

[rho, pval] = corr(ev_vect,vert_vect,'type','Kendall');
fprintf('Correlation between the EV RDM and the Vertebrae model is : %f \n', rho);
fprintf('p-value: %f \n', pval)


% For comparing the different primates, we can remove all the other species
% since there are no predictions for those.
% However, we also need to perform an additional step, which is to
% partition within each species, to create an additional relation for a 
% Within-Category correlation. 
%
% Over here, we split it by looking at the activation for each species in
% odd and even runs separately.
numvx = length(find(vt_m==1));
vtmat_odd = NaN(6,numvx);
vtmat_even = NaN(6,numvx);

% Odd runs RDM
for c = 1 : 6
    ct = 1;
    tmp = NaN(10,numvx);
    for i = c : 12 : 60
        tm = t(:,:,:,i);
        tmp(ct,:) = tm(logical(vt_m));
        ct = ct + 1;
    end
    vtmat_odd(c,:) = nanmean(tmp,1);
end

% Even runs RDM
for c = 1 : 6
    ct = 1;
    tmp = NaN(10,numvx);
    for i = c+6 : 12 : 60
        tm = t(:,:,:,i);
        tmp(ct,:) = tm(logical(vt_m));
        ct = ct + 1;
    end
    vtmat_even(c,:) = nanmean(tmp,1);
end

% We create a new matrix only for Monkey and Lemur, with the matrix in the
% following Order:
% 1- Monkey (Odd Runs)
% 2- Monkey (Even Runs)
% 3- Lemur (Odd Runs)
% 4- Lemur (Even Runs);

primate_vt_mat = [vtmat_odd(1,:);vtmat_even(1,:);
    vtmat_odd(2,:);vtmat_even(2,:)];

primate_vt_rdm = squareform(pdist(primate_vt_mat,'correlation'));
figure;imagesc(primate_vt_rdm);
title('Primate VT RDM')
labels={'monkey odd','monkey even' 'lemur odd','lemur even'}';
set(gca,'XTick',1:4,'XTickLabel',labels)
set(gca,'YTick',1:4,'YTickLabel',labels)

primate_rdm = [ 
    0 0 1 1;
    0 0 1 1;
    1 1 0 0;
    1 1 0 0
    ];

idx = logical(tril(ones(4,4),-1));
primate_vt_vect = primate_vt_rdm(idx);
primate_vect = primate_rdm(idx);

[rho, pval] = corr(primate_vt_vect,primate_vect,'type','Kendall');
fprintf('Correlation between the VT RDM and the Primate model is : %f \n', rho);
fprintf('p-value: %f \n', pval)

% One of the potential confound above, is that when comparing between
% Monkeys and Lemur, correlation can be inflated by the comparisons from
% the same run. To address this, we can also do a direct Between - Within
% species distance, excluding trials from the same run.
%
% A one sample t-test can then be conducted across subjects on the
% difference value, to examine if Between-category distance is greater than
% Within-category distance.

Within = [primate_vt_rdm(2,1);primate_vt_rdm(4,3)];
Btw =[primate_vt_rdm(3,2);primate_vt_rdm(4,1)];

Diff = mean(Btw) - mean(Within);


% If you are interested in running the entire process on a new subject,
% the data for a second subject is saved in
% /fMRI_data_Connolly2012/Subject02/
%
% 




