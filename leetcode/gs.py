# start point 0,0
# end point m-1,n-1

class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        self.count = 0
        self.walk(0,0,m-1,n-1)
        return self.count

    def walk(self,start_i,start_j,end_i,end_j):
        if start_i==end_i and start_j==end_j:
            self.count += 1
        
        if start_j+1 <= end_j:
            self.walk(start_i,start_j+1,end_i,end_j)
        
        if start_i+1 <= end_i:
            self.walk(start_i+1,start_j,end_i,end_j)
