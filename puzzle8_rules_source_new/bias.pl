max_vars(6).   % Max variables per clause
max_body(8).   % Max body literals per clause

head_pred(f, 1). 

body_pred(tile, 1).
body_pred(indx, 1).
body_pred(tile0, 1).
body_pred(tile1, 1).
body_pred(tile2, 1).
body_pred(tile3, 1).
body_pred(tile4, 1).
body_pred(tile5, 1).
body_pred(tile6, 1).
body_pred(tile7, 1).
body_pred(tile8, 1).
body_pred(indx1, 1).
body_pred(indx2, 1).
body_pred(indx3, 1).
body_pred(indx4, 1).
body_pred(indx5, 1).
body_pred(indx6, 1).
body_pred(indx7, 1).
body_pred(indx8, 1).
body_pred(indx9, 1).

body_pred(beforeto, 2).         
body_pred(nextto, 2).           
body_pred(above, 2).            
body_pred(adjacent_horiz, 2).   

body_pred(onrow, 3).
body_pred(valid_var, 1).

body_pred(after_tile, 2).       
body_pred(last_tile, 1).        

body_pred(goal, 1).             
body_pred(goal_index, 2).
body_pred(inplace_clause, 2).
body_pred(not_inplace_clause, 2).
body_pred(is_distinct, 2).
body_pred(distinct_indices, 2).

body_pred(inplace_from, 2).     

body_pred(row1_comp, 1).
body_pred(row2_comp, 1).
body_pred(row3_comp, 1).
body_pred(col1_comp, 1).
body_pred(col2_comp, 1).
body_pred(col3_comp, 1).



type(list).
type(tile).
type(indx).

type(f, (list,)). 

type(tile, (tile,)).
type(indx, (indx,)).
type(tile0, (tile,)).
type(tile1, (tile,)).
type(tile2, (tile,)).
type(tile3, (tile,)).
type(tile4, (tile,)).
type(tile5, (tile,)).
type(tile6, (tile,)).
type(tile7, (tile,)).
type(tile8, (tile,)).
type(indx1, (indx,)).
type(indx2, (indx,)).
type(indx3, (indx,)).
type(indx4, (indx,)).
type(indx5, (indx,)).
type(indx6, (indx,)).
type(indx7, (indx,)).
type(indx8, (indx,)).
type(indx9, (indx,)).

type(beforeto, (indx, indx)).
type(nextto, (indx, indx)).
type(above, (indx, indx)).
type(adjacent_horiz, (indx, indx)). 

type(onrow, (list, tile, indx)).
type(valid_var, (tile,)).

type(after_tile, (tile, tile)).    
type(last_tile, (tile,)).         

type(goal, (list,)).
type(goal_index, (tile, indx)).
type(inplace_clause, (list, tile)).
type(not_inplace_clause, (list, tile)).
type(is_distinct, (indx, indx)).
type(distinct_indices, (indx, indx)).

type(inplace_from, (list, tile)).  

type(row1_comp, (list,)).
type(row2_comp, (list,)).
type(row3_comp, (list,)).
type(col1_comp, (list,)).
type(col2_comp, (list,)).
type(col3_comp, (list,)).



direction(f, (in,)). % Input state to check

direction(tile, (out,)). 
direction(indx, (out,)). 
direction(tile0, (out,)).
direction(tile1, (out,)).
direction(tile2, (out,)).
direction(tile3, (out,)).
direction(tile4, (out,)).
direction(tile5, (out,)).
direction(tile6, (out,)).
direction(tile7, (out,)).
direction(tile8, (out,)).
direction(indx1, (out,)).
direction(indx2, (out,)).
direction(indx3, (out,)).
direction(indx4, (out,)).
direction(indx5, (out,)).
direction(indx6, (out,)).
direction(indx7, (out,)).
direction(indx8, (out,)).
direction(indx9, (out,)).

direction(beforeto, (in, in)).       
direction(nextto, (in, in)).         
direction(above, (in, in)).          
direction(adjacent_horiz, (in,in)). 

direction(onrow, (in, in, in)).      
direction(valid_var, (in,)).        

direction(after_tile, (in, out)).   
direction(last_tile, (in,)).        

direction(goal, (in,)).              
direction(goal_index, (in, out)).   
direction(inplace_clause, (in, in)). 
direction(not_inplace_clause, (in, in)). 
direction(is_distinct, (in, in)).
direction(distinct_indices, (in, in)).

direction(inplace_from, (in, in)).  

direction(row1_comp, (in,)).         
direction(row2_comp, (in,)).         
direction(row3_comp, (in,)).         
direction(col1_comp, (in,)).         
direction(col2_comp, (in,)).         
direction(col3_comp, (in,)).         