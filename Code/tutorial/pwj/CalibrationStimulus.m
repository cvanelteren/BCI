%% CalibrationStimulus.m 
%  -- a file which shows the stimulus for the calibration phase

%% change directory and initialize paths to all functions
try
    cd(fileparts(mfilename('fullpath')));
    catch
end;

try
   run ../../matlab/utilities/initPaths.m
   catch
   msgbox({'Please change to the directory where this file is saved before running the rest of this code'}, ...
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

%% set calibration parameters

% set the real-time-clock to use
initgetwTime; initsleepSec;

verb             = 0;
nSeq             = 10;         % number of sequences (and target cues) to present per calibration phase
nRepetitions     = 5;          % number of repetitions of stimulus set per sequence
cueDuration      = 2;          % the length a target character is presented, here: 2 s
stimDuration     = .1;         % the length a stimulus character is presented, here: 100 ms
interSeqDuration = 2;          % the length of an intersequence pause
bgCol            = [.5 .5 .5]; % background color (grey)
tgtCol           = [0 1 0];    % target indication color (green)
stimCol          = [0 0 0];    % stimulus color (black)
stimSize         = 25;         % set the stimulus size

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
     
% make the randomized target stimulus sequence, draw nSeq targets randomly
tgtSeq = symbols(randperm(length(symbols))); tgtSeq = tgtSeq(1:nSeq);

% send beginning of training marker
sendEvent('stimulus.training','start');

for si = 1:nSeq; % iterate over sequences
  
  % wait for interSeqDuration seconds between sequences & send marker
  sleepSec(interSeqDuration);
  sendEvent('stimulus.sequence','start');
  
  % present the target/cue stimulus 
  h = text(gca,.5,.5,tgtSeq(si),'fontunits','pixel','fontsize',stimSize, ...
         'HorizontalAlignment','center', 'VerticalAlignment','middle','FontWeight','bold','Color',tgtCol);
  
  % draw figure, send event and print on console
  drawnow;
  sendEvent('stimulus.targetSymbol',tgtSeq{si});
  fprintf('%d) tgt=%s : ',si,tgtSeq{si});
  
  % present the cue stimulus for cueDuration time
  sleepSec(cueDuration);  

  % reset cue stimulus to background color
  rectangle('Position',[0 0 1 1],'FaceColor',bgCol,'EdgeColor',bgCol)
  drawnow;
  
  % keep clear screen for 1 sec after cue presentation and before stimulus presentation
  sleepSec(1);  
 
  % present the set of stimuli for nRepetitions repetitions
  for ri = 1:nRepetitions
      
      % make the randomized stimulus sequence via randperm,
      % i.e. draw length(symbols) stimuli from the alphabet without replacement
      stimuli = symbols(randperm(length(symbols)));
      
      % present the alphabet in randomized order
      for ei = 1:length(symbols);
        
        % present the stimulus 
        h = text(gca,.5,.5,stimuli(ei),'fontunits','pixel','fontsize',stimSize, ...
         'HorizontalAlignment','center', 'VerticalAlignment','middle','FontWeight','bold','Color',stimCol);
        drawnow;
        
        ev        = sendEvent('stimulus.stimShow',stimuli{ei});           % indicate this stimulus is shown
        wasTarget = strcmp(stimuli(ei),tgtSeq(si));                       % compare stimulus to target
        sendEvent('stimulus.tgtShow',wasTarget,ev.sample);                % indicate if it was a 'target' shown
        
        % present the stimulus for stimDuration time
        sleepSec(stimDuration);
        
        % reset cue stimulus to background color        
        rectangle('Position',[0 0 1 1],'FaceColor',bgCol,'EdgeColor',bgCol)
        drawnow;
        
      end
  end
   
  % reset cue and fixation point to indicate end of sequence
  % set(h(:),'color',bgCol); drawnow;
  sendEvent('stimulus.sequence','end'); 
  
end 

% send end of training marker
sendEvent('stimulus.training','end');

%         % make the stimulus, i.e. a big circle in the middle of the axes
%         clf;
%         set(gcf,'color',[0 0 0],'toolbar','none','menubar','none'); % black figure
%         set(gca,'visible','off','color',[0 0 0]); % black axes
%         h=rectangle('curvature',[1 1],'position',[.25 .25 .5 .5],'facecolor',[.5 .5 .5]);
%         set(h,'visible','off');
%         % update the circles color
%         set(h,'color',[1 1 1],'visible','on'); % make it white and visible