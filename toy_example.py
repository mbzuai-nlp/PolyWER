from polywer_algorithm import polywer_multi, cleanList
from jiwer import wer, cer
import evaluate 

cer_threshold = 0.25
cosine_threshold = 0.85


rs = []
rs.append([
        '[Biochemistry, pharmacology] وهني [physics] و [Materials] والأمور هاي.', 
        '[بايوكيميستري، فارماكولوجي] وهني [فيزيكس] و [ماتيريالز] والأمور هاي.',
        '[الكيمياء الحيوية، الصيدلة] وهني [الفيزياء] و [المواد] والأمور هاي.'
])

rs.append([
        '[My passion was architecture] من البداية، أنا كنت أعرف هالشي', 
        '[ماي باشون واز أركيتكتشور] من البداية، أنا كنت أعرف هالشي',
        '[كان شغفي هو الهندسة المعمارية] من البداية، أنا كنت أعرف هالشي'
])

hs = [
    'biochemistry فارماكولوجي وهنى الفيزياء و materials والأمور هاي',
    'My passion واز أركيتكتشور من البداية أنا كنت أعرف هالشي'
]

############################ POLYWER ############################

poly_result = polywer_multi(rs, hs, cer_threshold, cosine_threshold)

############################ WER, CER ############################
rs = [cleanList(item) for item in rs]
hs = [cleanList(item) for item in hs]

wer_result = wer(rs, hs)
cer_result = cer(rs, hs)

############################ BLEU ############################

bleu = evaluate.load('bleu')
bleu_result = bleu.compute(references=rs, predictions=hs)['bleu']

############################ BERTScore ############################

bertscore = evaluate.load('bertscore')
bertscore_result = bertscore.compute(references=rs, predictions=hs, model_type="google/mt5-large")['f1'][0]

############################ mrWER ############################

'''
        Clone the repo https://github.com/qcri/multiRefWER and run:
        <mrwer_path>/mrwer.py -e <polywer_path>/ref_og <polywer_path>/ref_lit <polywer_path>/ref_lat <polywer_path>/hyp 

        Please note that we had to modify the mrwer code slightly to be able to run it (adding parentheses to the print statements and commenting out sys.reload)
'''

print(f'PolyWER: {poly_result}\nWER: {wer_result}\nCER: {cer_result}\nBLEU: {bleu_result}\nBertScore: {bertscore_result}')