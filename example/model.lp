\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize multi-objectives
  buffer_size: Priority=0 Weight=1 AbsTol=1e-06 RelTol=0
   192512 C14 + 192512 C15 + 192512 C16 + 192512 C17 + 192512 C18 + 9072 C19
  throughput: Priority=1 Weight=1 AbsTol=1e-06 RelTol=0
   - t
Subject To
 R0: - 17079 t - C1 + C5 >= 0
 R1: - 17079 t >= -1
 R2: - 17079 t - C1 + C8 >= 1
 R3: - 17079 t - C1 + C13 >= 1
 R4: - 13824 t - C2 + C3 >= 1
 R5: - 13824 t >= -1
 R6: - 13824 t - C2 + C9 >= 1
 R7: - 13824 t - C2 + C12 >= 1
 R8: - 13824 t - C3 + C4 >= 1
 R9: - 13824 t >= -1
 R10: - 13824 t - C3 + C9 >= 1
 R11: - 13824 t - C3 + C10 >= 1
 R12: - 13824 t - C4 + C6 >= 0
 R13: - 13824 t >= -1
 R14: - 13824 t - C4 + C10 >= 1
 R15: - 13824 t - C4 + C11 >= 1
 R16: - 3008 t + C2 - C5 >= 1
 R17: - 3008 t >= -1
 R18: - 3008 t - C5 + C8 >= 0
 R19: - 3008 t - C5 + C12 >= 0
 R20: - 3008 t >= -1
 R21: - 3008 t - C6 + C11 >= 0
 R22: - 142 t + C1 - C7 >= 0
 R23: - 142 t >= -1
 R24: C1 - C8 + C14 >= 0
 R25: C2 - C9 + C15 >= 0
 R26: C3 - C10 + C16 >= 0
 R27: C4 - C11 + C17 >= 0
 R28: C5 - C12 + C18 >= 0
 R29: C7 - C13 + C19 >= 0
 R30: - 13824 t - C2 + C3 + C20 >= 0
 R31: - 13824 t + C2 - C3 - C20 >= -1
 R32: - 13824 t - C2 + C4 + C21 >= 0
 R33: - 13824 t + C2 - C4 - C21 >= -1
 R34: - 13824 t - C3 + C4 + C22 >= 0
 R35: - 13824 t + C3 - C4 - C22 >= -1
 R36: - 3008 t - C6 + C7 + C23 >= 0
 R37: - 142 t + C6 - C7 - C23 >= -1
 R38: 41472 t <= 1
 R39: 17079 t <= 1
 R40: 3150 t <= 1
 R41: 3008 t <= 1
 R42: <= 1
Bounds
 C14 >= 2
 C15 >= 3
 C16 >= 3
 C17 >= 2
 C18 >= 3
 C19 >= 2
 C20 free
 C21 free
 C22 free
 C23 free
Generals
 C14 C15 C16 C17 C18 C19 C20 C21 C22 C23
End
