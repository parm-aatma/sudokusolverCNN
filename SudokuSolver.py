from keras.models import load_model
import cv2 as cv
import math
import numpy as np
import operator
import time
class Sudoku:
    def __init__(self):
        self.model=load_model('C:/Users/shiva/Desktop/2.0/sudoku_manualmodel.hdf5')

    def generateMatrix(self,sudoku):
        sudoku=cv.resize(sudoku,(640,740))
        original=sudoku.copy()



        blur=cv.GaussianBlur(sudoku,(9,9),0)

        thresh=cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)

        bitwise = cv.bitwise_not(thresh, thresh)

        kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]],np.uint8)

        dilated = cv.dilate(thresh, kernel)

        contours,_ = cv.findContours(dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv.contourArea, reverse=True)
        polygon = contours[0]
        #cv.drawContours(sudoku, contours[0], -1, (0, 255, 0), 3)
        #cv.imshow('as',sudoku)
        bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
        bottom_right=polygon[bottom_right][0]
        top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
        top_left=polygon[top_left][0]
        bottom_left, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
        bottom_left=polygon[bottom_left][0]
        top_right, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
        top_right=polygon[top_right][0]

        cv.circle(sudoku,tuple(top_left),5,(0,0,255),-1,1)
        cv.circle(sudoku,tuple(top_right),5,(0,0,255),-1,1)
        cv.circle(sudoku,tuple(bottom_left),5,(0,0,255),-1,1)
        cv.circle(sudoku,tuple(bottom_right),5,(0,0,255),-1,1)

        def distance_between(pt1,pt2):
            return math.sqrt(pow(pt1[0]-pt2[0],2)+math.pow(pt1[1]-pt2[1],2))

        #top_left, top_right, bottom_right, bottom_left = crop_rect[0], crop_rect[1], crop_rect[2], crop_rect[3]
        src = np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')
        side = max([  distance_between(bottom_right, top_right),
                    distance_between(top_left, bottom_left),
                    distance_between(bottom_right, bottom_left),
                    distance_between(top_left, top_right) ])

        dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype='float32')
        m = cv.getPerspectiveTransform(src, dst)
        sudoku=cv.warpPerspective(sudoku, m, (int(side), int(side)))
        sudoku=cv.adaptiveThreshold(sudoku, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
        #cv.imshow('original',sudoku)

        squares = []
        side = sudoku.shape[:1]
        side = side[0] / 9
        for j in range(9):
            for i in range(9):
                p1 = (i * side, j * side)  #Top left corner of a box
                p2 = ((i + 1) * side, (j + 1) * side)  #Bottom right corner
                squares.append((p1, p2))
        matrix=[]
        i=0
        empty=[]
        kernel2=np.ones((3,3))
        for square in squares:
            pt1=int(square[0][0])
            pt2=int(square[0][1])
            pt3=int(square[1][0])
            pt4=int(square[1][1])
            sq=sudoku[pt1:pt3,pt2:pt4]
            sq=cv.resize(sq,(28,28))
            sq=cv.bitwise_not(sq,sq)
            #sq = cv.GaussianBlur(sq, (3, 3), 0)
            sq = cv.dilate(sq, kernel2)
            #sq=cv.erode(sq,kernel)
            #cv.imshow('image',sq)

            #cv.waitKey(0)
            #break
            sq=sq.reshape((1,28,28,1))
            pred=self.model.predict(sq).tolist()
            pred=self.model.predict_classes(sq)
            if pred==0:
                matrix.append(0)
                empty.append(i)
            else:
                matrix.append((pred-1)[0])
            i+=1
        final=[]
        
        for i in range(0,81,9):
            temp=[]
            for j in range(i,i+9):
                print(i,end='\r')
                temp.append(matrix[j])
            final.append(temp)
        final=np.array(final).T
        final=final.tolist()
        return final,sudoku,squares,empty
    def Solve(self,board):
        subtract_set = {1,2,3,4,5,6,7,8,9}

        def check_horizontal(i,j):
            return subtract_set - set(container[i])

        def check_vertical(i,j):
            ret_set = []
            for x in range(9):
                ret_set.append(container[x][j])
            return subtract_set - set(ret_set)

        def check_square(i,j):
            first = [0,1,2]
            second = [3,4,5]
            third = [6,7,8]
            find_square = [first,second,third]
            for l in find_square:
                if i in l:
                    row = l
                if j in l:
                    col = l
            ret_set = []
            for x in row:
                for y in col:
                    ret_set.append(container[x][y])
            return subtract_set - set(ret_set)

        def get_poss_vals(i,j):
            poss_vals = list(check_square(i,j).intersection(check_horizontal(i,j)).intersection(check_vertical(i,j)))
            return poss_vals

        def explicit_solver(container):
            stump_count = 1
            for i in range(9):
                for j in range(9):
                    if container[i][j] == 0:
                        poss_vals = get_poss_vals(i,j)
                        if len(poss_vals) == 1:
                            container[i][j] = list(poss_vals)[0]
                            #print_container(container)
                            stump_count = 0
            return container, stump_count

        def implicit_solver(i,j,container):
            if container[i][j] == 0:
                poss_vals = get_poss_vals(i,j)
                
                #check row
                row_poss = []
                for y in range(9):
                    if y == j:
                        continue
                    if container[i][y] == 0:
                        for val in get_poss_vals(i,y):
                            row_poss.append(val)
                if len(set(poss_vals)-set(row_poss)) == 1:
                    container[i][j] = list(set(poss_vals)-set(row_poss))[0]
                    #print_container(container)
                
                #check column
                col_poss = []
                for x in range(9):
                    if x == i:
                        continue
                    if container[x][j] == 0:
                        for val in get_poss_vals(x,j):
                            col_poss.append(val)
                if len(set(poss_vals)-set(col_poss)) == 1:
                    container[i][j] = list(set(poss_vals)-set(col_poss))[0]
                    #print_container(container)

        def explicit_solver(container):
            stump_count = 1
            for i in range(9):
                for j in range(9):
                    if container[i][j] == 0:
                        poss_vals = get_poss_vals(i,j)
                        if len(poss_vals) == 1:
                            container[i][j] = list(poss_vals)[0]
                            #print_container(container)
                            stump_count = 0
            return container, stump_count

        def implicit_solver(i,j,container):
            if container[i][j] == 0:
                poss_vals = get_poss_vals(i,j)
                
                #check row
                row_poss = []
                for y in range(9):
                    if y == j:
                        continue
                    if container[i][y] == 0:
                        for val in get_poss_vals(i,y):
                            row_poss.append(val)
                if len(set(poss_vals)-set(row_poss)) == 1:
                    container[i][j] = list(set(poss_vals)-set(row_poss))[0]
                    #print_container(container)
                
                #check column
                col_poss = []
                for x in range(9):
                    if x == i:
                        continue
                    if container[x][j] == 0:
                        for val in get_poss_vals(x,j):
                            col_poss.append(val)
                if len(set(poss_vals)-set(col_poss)) == 1:
                    container[i][j] = list(set(poss_vals)-set(col_poss))[0]
                    #print_container(container)
                return container
        def print_container(container):
            for i, row in enumerate(container):
                for j, val in enumerate(row):
                    if (j)%3 == 0 and j<8 and j>0:
                        print("|",end=' ')
                    print(val,end=' ')
                print()
                if (i-2)%3 == 0 and i<8:
                    print("_____________________", end='')
                    print()
                print()
            print()
            print("||||||||||||||||||||||")
            print()
        container=board
        print(len(container))
        start = time.time()
        zero_count = 0
        for l in container:
            for v in l:
                if v == 0:
                    zero_count += 1
                    
        print(f'There are {zero_count} moves I have to make!')
        print()

        print_container(container)
        print()
        solving = True


        while solving:
            #Solver Portion
            container, stump_count = explicit_solver(container)
            
            #Loop-Breaking Portion
            zero_count = 0
            for l in container:
                for v in l:
                    if v == 0:
                        zero_count += 1
            if zero_count==0:
                solving=False
            if stump_count > 0:
                for i in range(9):
                    for j in range(9):
                        container = implicit_solver(i,j,container)
        print()
        return container
        print('That took', time.time()-start, 'seconds!')


    def fillBoard(self,board,squares,ans,empty):
        j=0
        for square in squares:  
            if  j in empty:
                pt1=int(square[0][0])
                pt2=int(square[0][1])
                pt3=int(square[1][0])
                pt4=int(square[1][1])
                sq=board[pt1:pt3,pt2:pt4]
                sq = cv.putText(sq,str(ans[j]) ,(26,62), cv.FONT_HERSHEY_SIMPLEX, 2, (127,127,0),4, 2)
                board[pt1:pt3,pt2:pt4]=sq 
            j+=1
        return board