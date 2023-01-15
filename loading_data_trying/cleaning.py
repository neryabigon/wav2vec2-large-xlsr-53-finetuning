import pyarabic.araby as araby
from unidecode import unidecode

# before_filter = "وَرَوَى حُمَيْدُ عَنْ أَنَاسٍ أَنَّ النَّبِيَّ صَلَّى اللَّهُ عَلَيْهِ وَسَلَّمَ قَالَ"
# after_filter = araby.strip_diacritics(before_filter)

before_filter = 'a atuação do vietnã na crise do coronavírus foi satisfatória'
after_filter = unidecode(before_filter)

print("Before clean: " + before_filter)
print("After clean: " + after_filter)
