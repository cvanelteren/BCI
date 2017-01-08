%% TrainingSignals.m 
%  -- a file which contains the code for training the classifier based on the saved calibration data

%% change directory and initialize paths to all functions

try
    cd(fileparts(mfilename('fullpath')));
    catch
end;

try
   run ../../matlab/utilities/initPaths.m
catch
   msgbox({'Please change to the directory where this file is saved before running the rest of this code'},...
   'Change directory'); 
end

% N.B. only really need the header to get the channel information, and sample rate
buffhost='localhost'; buffport=1972; hdr=[];
% wait for the buffer to return valid header information, loop WHILE until ready
while ( isempty(hdr) || ~isstruct(hdr) || (hdr.nchans==0) )
  try 
    hdr=buffer('get_hdr',[],buffhost,buffport); 
  catch
    hdr=[];
    fprintf('Invalid header info... waiting.\n');
  end;
  pause(1);
end;

%% run the classifier training based on the acquired calibration data

% set the configuration for the used EEG cap, notice:
% capFile channel names override those from the header!
capFile      = 'cap_tmsi_mobita_im.txt';
overridechnm = 1;
fname        = 'clsfr';

% load the calibration data
dname        = 'calibrate_data';
load(dname);

% train classifier
clsfr = buffer_train_erp_clsfr(data,devents,hdr, ...
        'spatialfilter','slap','freqband',[0 .3 10 12],'badchrm',0, ...
        'capFile',capFile,'overridechnms',overridechnm);

% save result
fprintf(1,'Saving clsfr to : %s',fname);
save(fname,'-struct','clsfr');