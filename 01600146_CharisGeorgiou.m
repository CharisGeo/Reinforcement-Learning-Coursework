function [NumStates, NumActions, TransitionMatrix, RewardMatrix, StateNames, ActionNames, AbsorbingStates,gamma,tol,p,T,R] = RunCoursework()

[NumStates, NumActions, TransitionMatrix, RewardMatrix, StateNames, ActionNames, AbsorbingStates]= PersonalisedGridWorld(0.7);
T = TransitionMatrix;
R = RewardMatrix;
p=0.7;
gamma = 0.5;
tol = 0.0001;

[UnbiasedPolicy] = GetUnbiasedPolicy(AbsorbingStates,NumActions);

[V] = PolicyEvaluation(UnbiasedPolicy, TransitionMatrix, RewardMatrix, AbsorbingStates, gamma, tol)

[prob_start_state1,prob_start_state2,prob_start_state3] = SequenceProbabilities(UnbiasedPolicy,TransitionMatrix)

[improved_prob_start_state1,improved_prob_start_state2,improved_prob_start_state3] = DeterministicProbabilities(TransitionMatrix)

[TraceList,FirstVisitMC,Normal] = GenerateTraces(UnbiasedPolicy, TransitionMatrix, RewardMatrix, AbsorbingStates,ActionNames,NumActions,NumStates, gamma, tol,StateNames)


%% Question 2
%Based on Lab Assignment 1- Dr Aldo Faisal, Machine Learning and Neural Computation (2018-2019)
function [UnbiasedPolicy] = GetUnbiasedPolicy(AbsorbingStates, NumActions)
UnbiasedPolicy = 1./NumActions * ~AbsorbingStates'*ones(1,NumActions);%Generate an unbiased policy matrix

function [V] = PolicyEvaluation(UnbiasedPolicy, TransitionMatrix, RewardMatrix, AbsorbingStates, gamma, tol)
% Dynamic Programming: Policy Evaluation. Estimates V(s) for each state s.
% Using 2 vectors for keeping track of Value Function, V(s)
S = length(UnbiasedPolicy); % number of states - introspecting transition matrix
A = length(UnbiasedPolicy(1,:)); % number of actions - introspecting policy matrix
V = zeros(S, 1); % optimal value function vector 11x1 (V at step i)
newV = V; % (V at step i+1)
Delta = 2*tol; % ensure initial Delta is greater than tolerance
while Delta > tol % keep approximating while not met the tolerane level
    for priorState = 1 : S
        if AbsorbingStates(priorState) % do not update absorbing states
            continue;
        end
        tmpV = 0;%temporary value of value function
        for action = 1 : A
            tmpQ = 0;%temporary state-action value 
            for postState = 1 : S
                tmpQ = tmpQ + TransitionMatrix(postState,priorState,action)*(RewardMatrix(postState,priorState,action) + gamma*V(postState));%calculate new state-action value
            end
            tmpV = tmpV + UnbiasedPolicy(priorState,action)*tmpQ;%calculate new value function
        end
        newV(priorState) = tmpV;%update value function for each state
    end
    diffVec = abs(newV - V);%find the difference between new and old value functions
    Delta = max(diffVec);%update delta to be the max difference between the old and new value functions
    V = newV;
end
V = V([1,4:end])'; %display value functions excluding absorbing states

%% Question 3a
function [prob_start_state1,prob_start_state2,prob_start_state3] = SequenceProbabilities(UnbiasedPolicy,TransitionMatrix)
sequence1 = [14 10 8 4 3];%show sequence 1 in matrix form
 prob_start_state1 = 0.25;%probability to start at each of the states s11,s12,s13,s14
    for i = 1:length(sequence1)-1%for the number of transitions from beginning until the end of the sequence (i is taking each element of the sequence 1 row matrix sequentially)
      prior = sequence1(i);%prior state is equal to the i^th state
      post = sequence1(i+1);%post state is equal to the i^th+1 state
      prob_sequence1 = UnbiasedPolicy(prior,:)*squeeze(TransitionMatrix(post,prior,:));%find the probability of all transitions
      prob_start_state1= prob_start_state1 * prob_sequence1;%multiply the starting state probability with the probability of all transitions
    end

sequence2 = [11 9 5 6 6 2];%show sequence 2 in matrix form
 prob_start_state2 = 0.25;
    for i = 1:length(sequence2)-1
      prior = sequence2(i);
      post = sequence2(i+1);
      prob_sequence2 = UnbiasedPolicy(prior,:)*squeeze(TransitionMatrix(post,prior,:));
      prob_start_state2= prob_start_state2 * prob_sequence2;
    end

sequence3 = [12 11 11 9 5 9 5 1 2];%show sequence 3 in matrix form
 prob_start_state3 = 0.25;
    for i = 1:length(sequence3)-1
      prior = sequence3(i);
      post = sequence3(i+1);
      prob_sequence3 = UnbiasedPolicy(prior,:)*squeeze(TransitionMatrix(post,prior,:));
      prob_start_state3= prob_start_state3 * prob_sequence3;
    end

%% Question 3b
function [improved_prob_start_state1,improved_prob_start_state2,improved_prob_start_state3] = DeterministicProbabilities(TransitionMatrix)
%DeterministicActions  N   E   S   W
DeterministicPolicy = [0   1   0   0;%s1
                       0   0   0   0;%s2
                       0   0   0   0;%s3
                       0   0   1   0;%s4
                       0   1   0   0;%s5
                       0   1   0   0;%s6
                       0   0   0   1;%s7
                       0   0   0   1;%s8
                       1   0   0   0;%s9
                       1   0   0   0;%s10
                       1   0   0   0;%s11
                       0   0   0   1;%s12
                       0   0   0   1;%s13
                       1   0   0   0];%s14
      
 sequence1 = [14 10 8 4 3];
 improved_prob_start_state1 = 0.25;
    for i = 1:length(sequence1)-1
      prior = sequence1(i);
      post = sequence1(i+1);
      prob_sequence1 = DeterministicPolicy(prior,:)*squeeze(TransitionMatrix(post,prior,:));%find the probability of all transitions using the deterministic policy
      improved_prob_start_state1= improved_prob_start_state1 * prob_sequence1;
    end
     
  sequence2 = [11 9 5 6 6 2];
 improved_prob_start_state2 = 0.25;
    for i = 1:length(sequence2)-1
      prior = sequence2(i);
      post = sequence2(i+1);
      prob_sequence2 = DeterministicPolicy(prior,:)*squeeze(TransitionMatrix(post,prior,:));
      improved_prob_start_state2= improved_prob_start_state2 * prob_sequence2;
    end
    
 sequence3 = [12 11 11 9 5 9 5 1 2];
 improved_prob_start_state3 = 0.25;
    for i = 1:length(sequence3)-1
      prior = sequence3(i);
      post = sequence3(i+1);
      prob_sequence3 = DeterministicPolicy(prior,:)*squeeze(TransitionMatrix(post,prior,:));
      improved_prob_start_state3= improved_prob_start_state3 * prob_sequence3;
    end 
   
%% Question 4a
function [TraceList,FirstVisitMC,Normal] = GenerateTraces(UnbiasedPolicy, TransitionMatrix, RewardMatrix, AbsorbingStates,ActionNames,NumActions,NumStates, gamma, tol,StateNames)
N = 10;%Number of traces
StateVisited = [];%initialize empty array to store states visited for each trace
RewardsReceived = [];%initialize empty array to store rewards received
ActionTaken = {};%empty character array to store the actions taken
TraceList = {};%empty list to store states visited,actions taken and rewards received for each trace
for i = 1:N%for traces 1 to 10
     InitialState = 10 + randi(4);%pick a random starting state from s11 s12 s13 s14
     CharList='';%Character array constructor
     CurrentState = InitialState;%my initial state is now my current state
     k = 1;
     j = 1;
     while ~AbsorbingStates(CurrentState)%while the agent is not in an absorbing state
         Action = randi(4);%take an action 1=N 2=E 3=S 4=W
         possibleSuccessorState = find(TransitionMatrix(:,CurrentState,Action));%look which states are possible to go to next
         PostState = randsample(possibleSuccessorState,1,true,nonzeros(TransitionMatrix(:,CurrentState,Action)));%pick a successor state randomly
         Reward = RewardMatrix(PostState,CurrentState,Action);%agent gets reward for each move
         ActionTaken(i,j) = num2cell(ActionNames(Action));%save the action taken
         RewardsReceived(i,j) = Reward;%save the reward for each action
         if AbsorbingStates(PostState)%if the agent ends up in an absorbing state
             RewardsReceived(i,j) = RewardMatrix(PostState,CurrentState,Action);%get reward 0 for s2 and -10 for s3
         end
         StateVisited(i,j) = CurrentState;%save the successor state 
         CharList=[CharList ',' StateNames(CurrentState,:) ',' ActionNames(Action) ',' sprintf('%i',Reward)];%put the current state,action and reward into the character list array 
         CurrentState = PostState;%the successor state is now my current state  
         j = j + 1;
     end  
         CharList = CharList(2:end);%skip the first comma character
         TraceList{i,k}=CharList;%store all state,action reward sequences for all traces into the TraceList
         k = k+ 1;   
end

%% Question 4b 
Return = [];%Initialise an array to input the returns for each state for all traces
AverageReturns = [];%Initialise an array to input the average returns for each state for all traces
for tr_i = 1:N %trace1 trace2 trace3 trace4 trace5 trac6 trace7 trace8 trace9 trace10
    
    TraceLength = length(nonzeros(StateVisited(tr_i,:))); %find the length of the trace
    
    for st_j = 1:NumStates % s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11 s12 s13 s14
       % if ~AbsorbingStates(st_j)%if we are not in an absorbing state
       FirstState = find(StateVisited(tr_i,:) == st_j, 1, 'first'); % find each state separately from the first row of the first episode 
       Return(tr_i,st_j) = RewardsReceived(tr_i,FirstState:TraceLength)*(gamma.^(0:TraceLength-FirstState))' + 0;
   end        
end
CompleteTracesReturn= zeros(10,14);%create a [10x14] zeros array
CompleteTracesReturn(1,:) = Return(1,:);%store the returns for the first trace
for i = 2:10 % for the rest of the traces
    for j = 1:14 % for each state
        if ~isempty(nonzeros(Return(1:i,j)))%if nonzero values  
            if ~isempty(Return(i,j))%if zero values 
                CompleteTracesReturn(i,j) = sum((Return(1:i,j)))/length(nonzeros(Return(1:i,j)));%average the returns and store them
            else
                CompleteTracesReturn(i,j) = CompleteTracesReturn(i-1,j);%store the previous returns
            end
        else
            CompleteTracesReturn(i,j) = Return(i,j);%store the same returns
        end
    end
end
FirstVisitMC = nonzeros(CompleteTracesReturn(10,:))';%The First-Visit MC values are the nonzero values from the last row of the CompleteTracesReturn matrix 
FirstVisitMC(isnan(AverageReturns))=0;%convert NaN to zero in case a state is never visited
%% Question 4c
V = [-1.3103,0,0,-4.7923,-1.8617,-1.7455,-4.3571,-2.7541,-1.9763,-2.1293,-1.9960,-1.9999,-2.0036,-2.0221];
Difference = zeros(10,14);
Euclidean = zeros(10,1);
for i = 1:10 % for every trace
    Difference(i,:) = V - CompleteTracesReturn(i,:);%difference between value functions(q2) and averaged returns
     Euclidean(i) = norm(Difference(i,:)); %returns the Euclidean norm or Euclidean length.
end

Normal = figure;
plot(Euclidean)
xlabel('Number of Traces')
ylabel('Euclidean Distance')
title('Euclidean Distance against Number of Traces')
grid on