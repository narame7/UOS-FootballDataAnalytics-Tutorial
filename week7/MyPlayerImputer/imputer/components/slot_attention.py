import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSlotAttention(nn.Module):
    def __init__(self, num_slots, dim, heads=4, dim_head=64, iters=3, hidden_dim=128, eps=1e-8):
        '''
        num_slots: 몇 개의 의미 덩어리(슬롯)를 만들지, 예: 5개면 actor, teammates, GK(team), opponent, GK(opponent) 등
        dim: 각 feature의 차원
        heads: 어텐션 헤드의 수
        dim_head: 각 헤드의 차원
        iters: 반복 횟수 (slot을 몇 번 refine할지)
        eps: 수치 안정성을 위한 작은 상수
        hidden_dim: MLP의 내부 차원 (더 큰 값으로 설정하면 표현력 향상)

        Slot -> 각각의 개체를 대표하는 latent representation
        '''
        super().__init__()
        self.num_slots = num_slots
        self.dim = dim
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5
        
        # 슬롯 초기화 파라미터 (로그 표준편차 사용으로 수치 안정성 향상)
        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.xavier_uniform_(self.slots_logsigma)  # 더 나은 초기화
        
        # 멀티헤드 어텐션을 위한 설정
        self.heads = heads
        self.dim_head = dim_head
        dim_inner = dim_head * heads
        
        # 입력과 슬롯 정규화 레이어
        self.norm_inputs = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)
        
        # 헤드 분할 및 병합 연산
        self.to_q = nn.Linear(dim, dim_inner)  # 쿼리 변환
        self.to_k = nn.Linear(dim, dim_inner)  # 키 변환
        self.to_v = nn.Linear(dim, dim_inner)  # 값 변환
        
        # 헤드 결합 후 차원 변환
        self.combine_heads = nn.Linear(dim_inner, dim)
        
        # GRU 셀
        self.gru = nn.GRUCell(dim, dim)
        
        # 더 큰 hidden_dim으로 MLP 생성
        hidden_dim = max(dim, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim)
        )
    
    def forward(self, x, num_slots=None):  # x: (B, N, D)
        B, N, D = x.shape
        n_s = num_slots if num_slots is not None else self.num_slots
        device, dtype = x.device, x.dtype
        
        # 슬롯 초기화
        mu = self.slots_mu.expand(B, n_s, -1)
        sigma = self.slots_logsigma.exp().expand(B, n_s, -1)
        slots = mu + sigma * torch.randn(mu.shape, device=device, dtype=dtype)
        
        # 입력 정규화 및 키/값 변환
        x = self.norm_inputs(x)
        k = self.to_k(x)  # (B, N, H*D_h)
        v = self.to_v(x)  # (B, N, H*D_h)
        
        # 헤드 분할
        k = k.view(B, N, self.heads, self.dim_head).transpose(1, 2)  # (B, H, N, D_h)
        v = v.view(B, N, self.heads, self.dim_head).transpose(1, 2)  # (B, H, N, D_h)
        
        for _ in range(self.iters):  # slot에 집중하는 entity가 점점 뚜렷해짐
            slots_prev = slots
            slots = self.norm_slots(slots)
            
            # 쿼리 변환 및 헤드 분할
            q = self.to_q(slots)  # (B, n_s, H*D_h)
            q = q.view(B, n_s, self.heads, self.dim_head).transpose(1, 2)  # (B, H, n_s, D_h)
            
            # 어텐션 계산 (헤드별로)
            attn_logits = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale  # (B, H, n_s, N)
            attn = F.softmax(attn_logits, dim=-1) + self.eps  # (B, H, n_s, N)
            attn = attn / attn.sum(dim=-1, keepdim=True)  # 정규화
            
            # 값과 어텐션 가중치 적용
            updates = torch.einsum('bhij,bhjd->bhid', attn, v)  # (B, H, n_s, D_h)
            
            # 헤드 결합
            updates = updates.transpose(1, 2).contiguous().view(B, n_s, -1)  # (B, n_s, H*D_h)
            updates = self.combine_heads(updates)  # (B, n_s, D)
            
            # GRU 기반 슬롯 업데이트
            slots = self.gru(
                updates.view(-1, D),
                slots_prev.view(-1, D)
            )
            slots = slots.view(B, n_s, D)
            slots = slots + self.mlp(self.norm_pre_ff(slots))  # Residual connection
        
        return slots  # (B, num_slots, D), 각 슬롯은 의미 있는 그룹을 요약한 벡터
    
class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, iters=3): 
        '''
        num_slots: 몇 개의 의미 덩어리(슬롯)를 만들지, 예: 5개면 actor, teammates, GK(team), opponent, GK(opponent) 등
        dim: 각 feature의 차원
        iters: 반복 횟수 (slot을 몇 번 refine할지)

        Slot -> 각각의 개체를 대표하는 latent representation
        '''
        super().__init__()
        self.num_slots = num_slots # 
        self.iters = iters
        self.scale = dim ** -0.5 # attention softmax 안정화를 위해 사용하는 scaling factor

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))  # slot 초기 평균
        self.slots_sigma = nn.Parameter(torch.randn(1, 1, dim))  # slot 초기 표준편차

        self.project_q = nn.Linear(dim, dim) # (B, num_slots, dim)
        self.project_k = nn.Linear(dim, dim) # (B, num_inputs, dim)
        self.project_v = nn.Linear(dim, dim) # (B, num_inputs, dim)

        self.gru = nn.GRUCell(dim, dim) # 각 slot을 반복적으로 업데이트할 때 사용
        self.mlp = nn.Sequential( 
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        ) # residual 구조에서 refinement

        self.norm_inputs = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def forward(self, x):  # x: (B, N, D)
        B, N, D = x.shape

        # (1, num_slots, D) → expand to (B, num_slots, D)
        mu = self.slots_mu.expand(B, self.num_slots, D)
        sigma = F.softplus(self.slots_sigma).expand(B, self.num_slots, D)
        slots = mu + sigma * torch.randn_like(mu)

        x = self.norm_inputs(x)
        k = self.project_k(x) # (B, N, D)
        v = self.project_v(x) # (B, N, D)

        for _ in range(self.iters): # slot에 집중하는 entity가 점점 뚜렷해짐
            slots_prev = slots
            slots_norm = self.norm_slots(slots)

            q = self.project_q(slots_norm)  # (B, num_slots, D)
            # query (slot)와 key (input) 간 내적을 통해 유사도 계산
            attn_logits = torch.einsum('bid,bjd->bij', q, k) * self.scale  # (B, num_slots, N)
            attn = F.softmax(attn_logits, dim=-1) + 1e-8  # (B, num_slots, N)
            attn = attn / attn.sum(dim=-1, keepdim=True)

            updates = torch.einsum('bjn,bnd->bjd', attn, v)  # (B, num_slots, D)

            # GRU-based slot update, 이전 슬롯 값과 새로운 정보(updates)를 합쳐 업데이트
            slots = self.gru(
                updates.view(-1, D),
                slots_prev.view(-1, D)
            )
            slots = slots.view(B, self.num_slots, D)
            slots = slots + self.mlp(self.norm_pre_ff(slots))  # Residual connection

        return slots  # (B, num_slots, D), 각 슬롯은 freeze_frame 내에서 의미 있는 그룹을 요약한 벡터