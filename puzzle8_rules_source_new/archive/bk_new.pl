tile(b).
tile(t1).
tile(t2).
tile(t3).
tile(t4).
tile(t5).
tile(t6).
tile(t7).
tile(t8).

tile0(b).
tile1(t1).
tile2(t2).
tile3(t3).
tile4(t4).
tile5(t5).
tile6(t6).
tile7(t7).
tile8(t8).

indx(idx1).
indx(idx2).
indx(idx3).
indx(idx4).
indx(idx5).
indx(idx6).
indx(idx7).
indx(idx8).
indx(idx9).

indx1(idx1).
indx2(idx2).
indx3(idx3).
indx4(idx4).
indx5(idx5).
indx6(idx6).
indx7(idx7).
indx8(idx8).
indx9(idx9).

beforeto(idx1, idx2). 
beforeto(idx2, idx3).
beforeto(idx3, idx4). 
beforeto(idx4, idx5).
beforeto(idx5, idx6).
beforeto(idx6, idx7). 
beforeto(idx7, idx8).
beforeto(idx8, idx9).


adjacent_horiz(idx1, idx2).
adjacent_horiz(idx2, idx3).
adjacent_horiz(idx4, idx5).
adjacent_horiz(idx5, idx6).
adjacent_horiz(idx7, idx8).
adjacent_horiz(idx8, idx9).


nextto_horiz(I1, I2) :- adjacent_horiz(I1, I2).
nextto_horiz(I1, I2) :- adjacent_horiz(I2, I1).


above(idx1, idx4). % idx1 is directly above idx4
above(idx2, idx5).
above(idx3, idx6).
above(idx4, idx7).
above(idx5, idx8).
above(idx6, idx9).


nextto_vert(I1, I2) :- above(I1, I2).
nextto_vert(I1, I2) :- above(I2, I1).


nextto(I1, I2) :- nextto_horiz(I1, I2).
nextto(I1, I2) :- nextto_vert(I1, I2).


onrow([Tile,_,_,_,_,_,_,_,_],Tile, idx1).
onrow([_,Tile,_,_,_,_,_,_,_],Tile, idx2).
onrow([_,_,Tile,_,_,_,_,_,_],Tile, idx3).
onrow([_,_,_,Tile,_,_,_,_,_],Tile, idx4).
onrow([_,_,_,_,Tile,_,_,_,_],Tile, idx5).
onrow([_,_,_,_,_,Tile,_,_,_],Tile, idx6).
onrow([_,_,_,_,_,_,Tile,_,_],Tile, idx7).
onrow([_,_,_,_,_,_,_,Tile,_],Tile, idx8).
onrow([_,_,_,_,_,_,_,_,Tile],Tile, idx9).


valid_var(T):- tile(T). % Checks if T is a valid tile identifier.


after_tile(t1, t2).
after_tile(t2, t3).
after_tile(t3, t4).
after_tile(t4, t5).
after_tile(t5, t6).
after_tile(t6, t7).
after_tile(t7, t8).

last_tile(t8).


goal([b, t1, t2, t3, t4, t5, t6, t7, t8]).


goal_index(Tile, GoalIndex) :-
    goal(GoalState),
    onrow(GoalState, Tile, GoalIndex).


inplace_clause(S, T) :-
    goal_index(T, GoalIndex),
    onrow(S, T, GoalIndex). % Tile T is at its goal index in state S.


not_inplace_clause(S, T) :-
    goal_index(T, I_goal),        % Find where T should be
    onrow(S, T, I_current),       % Find where T currently is
    distinct_indices(I_current, I_goal). % Check if current index is not the goal index


is_distinct(idx1, idx2).
is_distinct(idx1, idx3).
is_distinct(idx1, idx4).
is_distinct(idx1, idx5).
is_distinct(idx1, idx6).
is_distinct(idx1, idx7).
is_distinct(idx1, idx8).
is_distinct(idx1, idx9).
is_distinct(idx2, idx3).
is_distinct(idx2, idx4).
is_distinct(idx2, idx5).
is_distinct(idx2, idx6).
is_distinct(idx2, idx7).
is_distinct(idx2, idx8).
is_distinct(idx2, idx9).
is_distinct(idx3, idx4).
is_distinct(idx3, idx5).
is_distinct(idx3, idx6).
is_distinct(idx3, idx7).
is_distinct(idx3, idx8).
is_distinct(idx3, idx9).
is_distinct(idx4, idx5).
is_distinct(idx4, idx6).
is_distinct(idx4, idx7).
is_distinct(idx4, idx8).
is_distinct(idx4, idx9).
is_distinct(idx5, idx6).
is_distinct(idx5, idx7).
is_distinct(idx5, idx8).
is_distinct(idx5, idx9).
is_distinct(idx6, idx7).
is_distinct(idx6, idx8).
is_distinct(idx6, idx9).
is_distinct(idx7, idx8).
is_distinct(idx7, idx9).
is_distinct(idx8, idx9).


distinct_indices(I1, I2) :- is_distinct(I1, I2).
distinct_indices(I1, I2) :- is_distinct(I2, I1).


inplace_from(S, T) :-
    last_tile(T),           % Base case: T is the last tile in the sequence (t8)
    inplace_clause(S, T).   % Check if T itself is in place

inplace_from(S, T) :-
    \+ last_tile(T),         % Recursive step: T is not the last tile
    after_tile(T, T_next),   % Find the next tile in the sequence
    inplace_clause(S, T),    % Check if the current tile T is in place
    inplace_from(S, T_next). % Recursively check from the next tile onwards



row1_comp(S) :- inplace_clause(S, b), inplace_clause(S, t1), inplace_clause(S, t2). % Checks indices 1, 2, 3
row2_comp(S) :- inplace_clause(S, t3), inplace_clause(S, t4), inplace_clause(S, t5). % Checks indices 4, 5, 6
row3_comp(S) :- inplace_clause(S, t6), inplace_clause(S, t7), inplace_clause(S, t8). % Checks indices 7, 8, 9


col1_comp(S) :- inplace_clause(S, b), inplace_clause(S, t3), inplace_clause(S, t6). % Checks indices 1, 4, 7
col2_comp(S) :- inplace_clause(S, t1), inplace_clause(S, t4), inplace_clause(S, t7). % Checks indices 2, 5, 8
col3_comp(S) :- inplace_clause(S, t2), inplace_clause(S, t5), inplace_clause(S, t8). % Checks indices 3, 6, 9