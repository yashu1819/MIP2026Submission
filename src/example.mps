NAME          SMALL_MILP
ROWS
 N  OBJ
 L  CONSTR1
 E  CONSTR2
COLUMNS
    MARK0000  'MARKER'                 'INTORG'
    Y         OBJ       2.0            CONSTR1   1.0
    Y         CONSTR2   1.0
    MARK0001  'MARKER'                 'INTEND'
    X         OBJ       1.0            CONSTR1   1.0
    X         CONSTR2   3.0
RHS
    RHS1      CONSTR1   10.0           CONSTR2   15.0
BOUNDS
 LO BND1      X         0.0
 LI BND1      Y         0.0
 UP BND1      Y         5.0
ENDATA
