from polywer_algorithm import polywer, polywer_multi

cer_threshold = 0.25
cosine_threshold = 0.85

# Individual (ref,hyp) pair
r = [
        '[Biochemistry, pharmacology] وهني [physics] و [Materials] والأمور هاي.', 
        '[بايوكيميستري، فارماكولوجي] وهني [فيزيكس] و [ماتيريالز] والأمور هاي.',
        '[الكيمياء الحيوية، الصيدلة] وهني [الفيزياء] و [المواد] والأمور هاي.'
]

h = 'biochemistry فارماكولوجي وهنى الفيزياء و materials والأمور هاي'

print(polywer(r, h, cer_threshold, cosine_threshold))

# Multiple (ref,hyp) pairs

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

print(polywer_multi(rs, hs, cer_threshold, cosine_threshold))
