class Solution:
    def decodeString(self, s: str) -> str:

        if not s:
            return ""
        
        if s[0].isalpha():
            return s[0] + self.decodeString(s[1:])
        
        # number
        repeat = int(s[0])
        balance = 0
        for i in range(1,len(s)):
            if s[i] == '[':
                balance += 1
            if s[i] == ']':
                balance -= 1
            
            if balance == 0:
                return repeat * s[2:i] + self.decodeString(s[i+1:])



        
if __name__ == "__main__":
    s = "3[a2[c]]"
    print(Solution().decodeString(s))  # Output: "aaabcbc"
        