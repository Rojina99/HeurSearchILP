max_vars(6).
max_body(8).

% head and body
head_pred(f, 1).
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
body_pred(onrow,3).
body_pred(inplace_from,2).
body_pred(row1_comp, 1).
body_pred(row2_comp, 1).
body_pred(row3_comp, 1).
body_pred(row1_l, 1).
body_pred(row_c1,1).

% types
type(f, (list, )).
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
type(onrow,(list, tile, indx)).
type(inplace_from,(list, tile)).
type(row1_comp, (list,)).
type(row2_comp, (list,)).
type(row3_comp, (list,)).
type(row1_l, (list,)).
type(row_c1,(list,)).

% directions
direction(f, (in, )).
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
direction(onrow,(in, in, in)).
direction(inplace_from,(in, in)).
direction(row1_comp, (in,)).
direction(row2_comp, (in,)).
direction(row3_comp, (in,)).
direction(row1_l, (in,)).
direction(row_c1,(in,)).