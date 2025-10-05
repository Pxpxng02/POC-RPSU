import json
import pandas as pd

INPUT_FILE = "arxiv-metadata-oai-snapshot.json"
OUTPUT_FILE = "outputs/arxiv_cs_filtered.csv"

#เลือกบทความที่เป็น Computer Science (cs.*) และอยู่ในช่วงปี 2014-2024
#โดยสามารถกำหนดจำนวนบทความที่จะอ่านได้ (MAX_RECORDS) เช่น 500000 หรือ None = อ่านทั้งหมด
TARGET_PREFIX = "cs."
START_YEAR = 2014
END_YEAR = 2024
MAX_RECORDS = None  # กำหนดจำนวนบทความที่จะอ่าน เช่น 500000 หรือ None = อ่านทั้งหมด

# ฟังก์ชันหลักในการกรองบทความ
def filter_cs_papers(input_file, output_file):
    results = []

    with open(input_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if MAX_RECORDS and i >= MAX_RECORDS:
                break
            try:
                record = json.loads(line)
            except Exception as e:
                print(f"⚠️ Error parsing line {i}: {e}")
                continue

            categories = record.get("categories", "")
            if not categories:
                continue

            # 1. Check categories: ต้องมี cs.*
            if not any(cat.startswith(TARGET_PREFIX) for cat in categories.split()):
                continue

            # 2. Check year
            update_date = record.get("update_date", "")
            if len(update_date) >= 4:
                try:
                    year = int(update_date[:4])
                except:
                    continue
                if year < START_YEAR or year > END_YEAR:
                    continue
            else:
                continue

            # ถ้าผ่านเงื่อนไขทั้งหมด → เก็บข้อมูล
            results.append({
                "id": record.get("id", ""),
                "title": record.get("title", "").replace("\n", " ").strip(),
                "abstract": record.get("abstract", "").replace("\n", " ").strip(),
                "authors": record.get("authors", "").strip(),
                "categories": categories,
                "update_date": update_date
            })

            # Log progress ทุก 100,000 records
            if i % 100000 == 0 and i > 0:
                print(f"Processed {i} records, collected {len(results)} matching papers")

    # Save CSV
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"✅ Saved {len(df)} papers to {output_file}")


if __name__ == "__main__":
    filter_cs_papers(INPUT_FILE, OUTPUT_FILE)
