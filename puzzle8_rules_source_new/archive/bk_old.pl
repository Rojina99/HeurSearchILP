is_list([_,_,_,_,_,_,_,_,_]).

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
nextto(I1,I2):- beforeto(I2,I1).

above(idx1, idx4).
above(idx2, idx5).
above(idx3, idx6).
above(idx4, idx7).
above(idx5, idx8).
above(idx6, idx9).
below(I1, I2):- above(I2,I1).

onrow([Tile,_,_,_,_,_,_,_,_],Tile, idx1).
onrow([_,Tile,_,_,_,_,_,_,_],Tile, idx2).
onrow([_,_,Tile,_,_,_,_,_,_],Tile, idx3).
onrow([_,_,_,Tile,_,_,_,_,_],Tile, idx4).
onrow([_,_,_,_,Tile,_,_,_,_],Tile, idx5).
onrow([_,_,_,_,_,Tile,_,_,_],Tile, idx6).
onrow([_,_,_,_,_,_,Tile,_,_],Tile, idx7).
onrow([_,_,_,_,_,_,_,Tile,_],Tile, idx8). 
onrow([_,_,_,_,_,_,_,_,Tile],Tile, idx9). 

inplace_clause(S,T):- onrow(S,T,I),T=b, I=idx1.
inplace_clause(S,T):- onrow(S,T,I),T=t1, I=idx2.
inplace_clause(S,T):- onrow(S,T,I),T=t2, I=idx3.
inplace_clause(S,T):- onrow(S,T,I),T=t3, I=idx4.
inplace_clause(S,T):- onrow(S,T,I),T=t4, I=idx5.
inplace_clause(S,T):- onrow(S,T,I),T=t5, I=idx6.
inplace_clause(S,T):- onrow(S,T,I),T=t6, I=idx7.
inplace_clause(S,T):- onrow(S,T,I),T=t7, I=idx8.
inplace_clause(S,T):- onrow(S,T,I),T=t8, I=idx9.

largest_tile(t8).

before_tile(b,t1).
before_tile(t1,t2).
before_tile(t2,t3).
before_tile(t3,t4).
before_tile(t4,t5).
before_tile(t5,t6).
before_tile(t6,t7).
before_tile(t7,t8).
after_tile(T1, T2):- before_tile(T2,T1).

inplace_from(S,T):-  largest_tile(T), inplace_clause(S,T).
inplace_from(S,T):-  inplace_clause(S,T), after_tile(T1,T),inplace_from(S,T1).

row1_comp(S) :- inplace_from(S,t6). % bottom most row
row2_comp(S) :- inplace_from(S,t3). % second n bottom most row
row3_comp(S) :- inplace_from(S,b).  % top row

% Everything is inplace 
row1_l(S):- inplace_from(S,t8).

row_c1(S):- inplace_from(S,t5), inplace_clause(S,t2).