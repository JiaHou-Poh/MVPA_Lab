%% ------------------ Learning Objectives ------------------ 
% The aim for this lab session is to develop an intuition of what MVPA
% does, and also to provide an introduction of different methods that are
% commonly used for MVPA.
%
% We will first be looking at classification approaches (Logistics
% regression and SVM), followed by the analysis of relational structures
% (RSA). 
%
% The lab session will be broken down into 3 sections as listed below, 
% and each section will be done within the individual m-files. 
%
% For ease of understanding, several of the functions used for the 
% exercises have been edited, and may be different from how they are 
% commonly implemented in popular packages (e.g. The logistics regression
% here uses a gradient descent. This is computationally inefficient, but if
% you are interested, you can plot how the cost function is minimised
% across the search).
%
% I tried including comments for most of the steps, and have written/edited
% the code to be as human readable as possible, but the implementations
% might not be as rigorous/efficient. While I have made sure that they
% produce similar results to external packages (in our test cases), it is
% (highly) likely that there may be bugs, so I strongly recommend that 
% you DO NOT use this for your own analysis and stick to the established
% packages!
% 
%
% In most instances, you only need to execute the written code. 
% This can be done either by 1)copying+pasting to the Command window, or 
% 2)Highlighting and evaluating the code (right click and select).
%
% You will not need to do (much) Matlab coding for this session, 
% but there are exercises embedded within, to help you better grasp the 
% relevant concepts (hopefully).
%
% By the end of this lab session, you should be able to :
% 1) Understand the general principles and approaches in MVPA
% 2) Identify the types of questions that can be addressed with MVPA
% 3) Comfortably implement basic linear classification and understanding
%    what regularization does.
% 4) Comfortably implement basic RSA and understanding how different models
%    can be created.
%
%
% Edited by JH POH - 2/16/2020

%% -------------- Code for Class --------------
% Logistcs Regression
open('LogisticsRegression_Exercise.m');

% SVM
open('SVM_Exercise.m');

% RSA
open('RSA_Exercise.m');

