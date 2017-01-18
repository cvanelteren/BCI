%% FeedbackSignals.m 
%  -- a file which contains the code for applying the trained classifier during the feedback stage

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
    hdr = buffer('get_hdr',[],buffhost,buffport); 
  catch
    hdr = [];
    fprintf('Invalid header info... waiting.\n');
  end;
  pause(1);
end;

%% apply the trained classifier within the feedback stage

% set the real-time-clock to use
initgetwTime;
initsleepSec;

% record 600ms of data (expected P300 duration)
verb     = 1;
trlen_ms = 600;

% load the trained classifier
cname    = 'clsfr';
clsfr    = load(cname);

if(isfield(clsfr,'clsfr')) 
clsfr    = clsfr.clsfr;
end;

state    = []; 
endTest  = 0; 
fs       = 0;

while ( endTest==0 )
    
  % reset the sequence information
  endSeq = 0; 
  fs(:)  = 0; % predictions
  nShow  = 0; % number presentations processed
  
  while ( endSeq==0 && endTest==0 )
      
    % make the buffer wait for data (to apply the classifier to) for trlen_ms
    [data,devents,state] = buffer_waitData(buffhost,buffport,state, ...
                           'startSet',{{'stimulus.stimShow'}}, ...
                           'trlen_ms',trlen_ms,'exitSet',{'data' {'stimulus.sequence' 'stimulus.feedback'} 'end'});
  
    %% process the events - get stimShow events, apply classifier and store predictions
    
    for ei=1:numel(devents) % iterate through received events devents
        
      if ( matchEvents(devents(ei),'stimulus.sequence','end') )     % end sequence
        endSeq= ei;                                                 % record which is the end-seq event
        
      elseif (matchEvents(devents(ei),'stimulus.feedback','end') )  % end training
        endTest = ei;                                               % record which is the end-feedback event
        
      elseif ( matchEvents(devents(ei),'stimulus.stimShow') )       % flash, apply the classifier
        
        if ( verb>0 ) fprintf('Processing event: %s',ev2str(devents(ei))); end;
        
        % apply classification pipeline to this events data
        [f,fraw,p] = buffer_apply_erp_clsfr(data(ei).buf,clsfr);
        
        % store the set of all predictions so far
        nShow                = nShow + 1;
        fs(1:numel(f),nShow) = f;
        
        if ( verb>0 ) fprintf(' = %g',f); end;
        
      end
    end
    
  end
  
  if ( endSeq>0 ) % send the accumulated predictions
    sendEvent('classifier.prediction',fs(:,1:nShow), devents(endSeq).sample);
  end
  
end