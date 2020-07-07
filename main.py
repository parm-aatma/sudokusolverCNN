from pandeys import SudokuSolver # this is the module in which the Sudoku Class is stored
import cv2 as cv
import numpy as np

ss=SudokuSolver.Sudoku()
image=cv.imread(' ',0) #put the image's path here

final,wrapped,squares,empty=ss.generateMatrix(image)

ans=ss.Solve(final)
ans=np.array(ans).T
ans=ans.tolist()
ravel=[]

for row in ans:
    ravel+=row
print(ravel)

filled=ss.fillBoard(wrapped,squares,ravel,empty)
print(empty)
image=cv.resize(image,(540,620))
cv.imshow('puzzle',image )
cv.imshow('solved',filled)
cv.waitKey(0)