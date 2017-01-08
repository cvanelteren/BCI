%% FeedbackStimulus.m 
%  -- a file which shows the stimulus and feedback for the feedback phase of the experiment

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

%% set feedback parameters

% set the real-time-clock to use
initgetwTime; initsleepSec;

verb         = 1;
nSeq         = 10;         % number of test sequences to run per testing phase
nRepetitions = 5;          % number of repetitions of stimulus set per sequence
cueDuration  = 2;          % the length a target character is presented, here: 2 s
stimDuration = .1;         % the length a stimulus character is presented, here: 100 ms
bgCol        = [.5 .5 .5]; % background color (grey)
tgtCol       = [0 0 1];    % predictor indication color (blue)
stimCol      = [0 0 0];    % stimulus color (black)
stimSize     = 25;         % set the stimulus size

dataDuration     = .6;     % lenght of the data that the sig processing needs
interSeqDuration = 2;      % length of time between sequences
feedbackDuration = 2;      % length of time feedback is on the screen
instructDuration = 2;      % length of time to show instruction

% create array of stimuli, get the alphabet by 
% converting integer values in the range [65,90] to ASCII strings
initChar = 65; alphabet = 26;
symbols  = cellstr( char(initChar:(initChar+alphabet-1))' ); 

% ----------------------------------------------------------------------------

% make the stimulus window: set background color, remove axis etc.
clf; 
set(gcf,'color',bgCol); 
set(gca,'visible','off');
set(gca,'YDir','reverse');
set(gca,'xlim',[0 1],'ylim',[0 1]);

% indicate beginning of feedback/testing session
sendEvent('stimulus.feedback','start');

for si = 1:nSeq;

  % at the beginning of each sequence, 
  % reset the counter of presented stimuli and stimulus sequences
  nShow = 0; stimSeq = zeros(size(symbols));
 
  % reset all symbols to background color %%%%%%%%%%%%%%%%%%
  % set(h(:),'color',bgCol);
  
  sleepSec(interSeqDuration);
  
  % present the stimulus
  h = text(gca,.5,.45,'Think of your target letter','fontunits','pixel','fontsize',20, ...
      'HorizontalAlignment','center', 'VerticalAlignment','middle','FontWeight','bold','Color',stimCol);
  h = text(gca,.5,.55,' and get ready','fontunits','pixel','fontsize',20, ...
      'HorizontalAlignment','center', 'VerticalAlignment','middle','FontWeight','bold','Color',stimCol);
  drawnow;
  
  sendEvent('stimulus.sequence','start');
  sleepSec(instructDuration);

  % reset instruction screen to background color        
  rectangle('Position',[0 0 1 1],'FaceColor',bgCol,'EdgeColor',bgCol)
  drawnow;
  
  sleepSec(instructDuration);
  
  % reset buffer_newevents to only return events matching type 'classifier.prediction'
  [~,state] = buffer_newevents(buffhost,buffport,[],'classifier.prediction',[],0);
    
  % present the set of stimuli for nRepetitions repetitions
  for ri = 1:nRepetitions
      
      % make the randomized stimulus sequence via randperm,
      % i.e. draw length(symbols) stimuli from the alphabet without replacement
      stimuli = symbols(randperm(length(symbols)));
  
      % present the alphabet in randomized order
      for ei = 1:length(symbols);
      
          % record the stimulus state, append second dimension (time) 
          %  - needed for decoding the classifier predictions later
          nShow  = nShow + 1;
          getIdx = cell2mat(stimuli(ei))-(initChar-1);
          stimSeq(getIdx, nShow) = true;

          % present the stimulus
          h = text(gca,.5,.5,stimuli(ei),'fontunits','pixel','fontsize',stimSize, ...
              'HorizontalAlignment','center', 'VerticalAlignment','middle','FontWeight','bold','Color',stimCol);
          drawnow;

          % indicate what stimulus was presented   
          sendEvent('stimulus.stimShow',stimuli{ei});

          % present the stimulus for stimDuration time
          sleepSec(stimDuration);

          % reset stimulus to background color        
          rectangle('Position',[0 0 1 1],'FaceColor',bgCol,'EdgeColor',bgCol)
          drawnow;
      end
  end
  
 sleepSec(1);
 %sleepSec(dataDuration-stimDuration);    % wait enough extra time for the last brain-response to finish
 sendEvent('stimulus.sequence','end');

  
  %% combine the classifier predictions with knowledge of the stimulus used
  
  % wait for the signal processing pipeline to return the set of predictions
  if( verb>0 ) fprintf(1,'Waiting for predictions\n'); end;
  
  % get predictions from classifier (stored in devents.value)
  [devents,state] = buffer_newevents(buffhost,buffport,state,'classifier.prediction',[],500);
  
  if ( ~isempty(devents) ) 
    
    pred = [devents.value]; % get all the classifier predictions in order
        
    % correlate the stimulus sequence stimSeq with the classifier predictions pred
    % to identify the most likely letter (predicted target has highest correlation)
    corr        = reshape(stimSeq(:,1:nShow), [numel(symbols) nShow]) * pred(:); 
    [~,predTgt] = max(corr);
      
    % present the classifier prediction
    h = text(gca,.5,.5, symbols(predTgt),'fontunits','pixel','fontsize',stimSize, ...
        'HorizontalAlignment','center', 'VerticalAlignment','middle','FontWeight','bold','Color',tgtCol);
    drawnow;
    sendEvent('stimulus.prediction',symbols{predTgt});
  end
  
  % present the predicted target stimulus for feedbackDuration time
  sleepSec(feedbackDuration);
  
  % reset predicted target stimulus to background color        
  rectangle('Position',[0 0 1 1],'FaceColor',bgCol,'EdgeColor',bgCol)
  drawnow;
end

% send end of testing/feedback marker
sendEvent('stimulus.feedback','end');