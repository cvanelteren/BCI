%% CalibrationSignals.m 
%  -- a file which does the signal processing associated with the calibration phase

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

%% run the actual signal processing associated with the calibration phase

% set the real-time-clock to use
initgetwTime;
initsleepSec;

% record 600ms of data (expected P300 duration)
verb     = 1;
trlen_ms = 600;
dname    = 'calibrate_data';

% call to buffer_waitData
[data,devents,state] = buffer_waitData(buffhost,buffport,[], ...
                       'startSet',{{'stimulus.tgtShow'}}, ...
                       'exitSet' ,{'stimulus.training' 'end'}, ...
                       'verb',verb,'trlen_ms',trlen_ms);

% remove the event indicating end of training
mi = matchEvents(devents,'stimulus.training','end'); 
devents(mi) = []; data(mi) = [];

% print to the console and save calibration data to file calibrate_data.mat
fprintf('Saving %d epochs to : %s\n',numel(devents),dname);
save('calibrate_data','data','devents');