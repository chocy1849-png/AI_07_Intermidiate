# PartA_RFP_AutoGrading_QBank v4 review report

- source file: (260417) PartA_RFP_AutoGrading_QBank_v3_fixed.json
- item count: 340

## Applied corrections
- missing `answer_format` fixed: 6
- leading/trailing whitespace in answer fixed: 1
- numbered multiple-choice answers canonicalized: 9
- inconsistent category labels normalized: 8
- `evidence.text` normalized to `evidence.quote`: 3
- explicit currency unit restored: 2

## answer_eval_type distribution
- choice: 233
- currency: 34
- string: 32
- duration: 15
- number: 6
- list: 6
- date: 5
- email: 4
- phone: 4
- number_pair: 1

## Notable high-confidence fixes
- DOC012_Q004: answer=`Tibero 6.0`, answer_eval_type=`string`, category=`specific`, choice_index=`None`
- DOC020_Q001: answer=`1,400,000,000원`, answer_eval_type=`choice`, category=`common`, choice_index=`4`
- DOC020_Q003: answer=`요구사항 정의 및 분석`, answer_eval_type=`choice`, category=`specific`, choice_index=`1`
- DOC020_Q004: answer=`9부`, answer_eval_type=`choice`, category=`specific`, choice_index=`4`
- DOC022_Q001: answer=`차장`, answer_eval_type=`choice`, category=`common`, choice_index=`2`
- DOC025_Q001: answer=`2024.4.15.(월) 10:30, GKL 본사 B1F 강당`, answer_eval_type=`choice`, category=`common`, choice_index=`1`
- DOC026_Q001: answer=`24시간 운전, 2인 1조 운영`, answer_eval_type=`choice`, category=`specific`, choice_index=`2`
- DOC026_Q002: answer=`인력관리 소홀 또는 조작 실수로 중대한 문제가 발생한 경우`, answer_eval_type=`choice`, category=`specific`, choice_index=`3`
- DOC026_Q003: answer=`제어시스템 또는 현장 장치를 활용하여 지속적으로 관찰`, answer_eval_type=`choice`, category=`specific`, choice_index=`2`
- DOC037_Q001: answer=`5,031,000,000원`, answer_eval_type=`choice`, category=`common`, choice_index=`3`
- DOC055_Q001: answer=`380,000,000원`, answer_eval_type=`currency`, category=`common`, choice_index=`None`
- DOC084_Q001: answer=`195,030,000원`, answer_eval_type=`currency`, category=`common`, choice_index=`None`
- DOC089_Q001: answer=`2024년 기초학문자료센터 시스템 운영 및 연구성과물 DB 구축`, answer_eval_type=`string`, category=`common`, choice_index=`None`
- DOC089_Q002: answer=`추진전략 및 수행계획과 이행방안 포함`, answer_eval_type=`choice`, category=`specific`, choice_index=`2`
- DOC089_Q003: answer=`계약일로부터 12개월`, answer_eval_type=`choice`, category=`common`, choice_index=`2`
- DOC093_Q003: answer=`보안모듈 소프트웨어 기능 구현`, answer_eval_type=`choice`, category=`specific`, choice_index=`3`
- DOC094_Q003: answer=`1:1 온라인 상담 게시판 기능 구축`, answer_eval_type=`choice`, category=`specific`, choice_index=`2`
- DOC096_Q003: answer=`업무전용앱 기능을 모바일오피스에 통합`, answer_eval_type=`choice`, category=`specific`, choice_index=`2`