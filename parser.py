
"""GNSS NMEA 파서: GGA, RMC, GSV 데이터를 CSV로 변환."""

import argparse
import csv
import re
import zipfile
import statistics

from collections import Counter
from pathlib import Path


# 1. NMEA 문장 검색
SENTENCE = re.compile(
    rb'\$([A-Z]{2}(?:GGA|RMC|GSV)),([^\r\n$]*?)\*([0-9A-Fa-f]{2})'
)


# 2. CSV에 저장할 항목
COLUMNS = [
    'source_file',
    'utc_time',
    'date_ddmmyy',
    'latitude_deg',
    'longitude_deg',
    'speed_kmh',
    'satellites_used',
    'hdop',
    'altitude_m',
    'fix_quality',
    'rmc_status',
    'gps_visible',
    'glonass_visible',
    'cn0_mean_dbhz',
    'cn0_max_dbhz',
    'cn0_count',
    'cn0_xsat_std_db'  # 추가: 위성 간 신호 세기 표준편차
]


# 3. 숫자 변환
def num(value):
    try:
        return float(value) if value else None
    except ValueError:
        return None


# 4. 위도 / 경도 변환
def coord(value, hemisphere):
    if not value or hemisphere not in ('N', 'S', 'E', 'W'):
        return None

    try:
        x = float(value)
        degrees = int(x // 100)
        minutes = x - degrees * 100

        if not (0 <= minutes < 60):
            return None

        result = degrees + minutes / 60

        limit = 90 if hemisphere in ('N', 'S') else 180

        if result > limit:
            return None

        return -result if hemisphere in ('S', 'W') else result

    except ValueError:
        return None


# 5. NMEA 데이터 파싱
def parse_bytes(raw, source):
    rows = []
    current = None
    gsv_pending = {}
    stats = Counter()

    for match in SENTENCE.finditer(raw):

        body = match.group(1) + b',' + match.group(2)

        checksum = 0
        for b in body:
            checksum ^= b

        if checksum != int(match.group(3), 16):
            stats['checksum_failed'] += 1
            continue

        stats['checksum_ok'] += 1

        kind = match.group(1)[2:].decode('ascii')
        talker = match.group(1)[:2].decode('ascii')

        p = match.group(2).decode(
            'ascii', 'replace'
        ).split(',')

        try:

            # RMC / GGA 처리
            if kind in ('RMC', 'GGA'):

                if not p[0]:
                    continue

                if (
                    current is None
                    or p[0] != current['utc_time']
                    or (
                        kind == 'RMC'
                        and current.get('_seen_rmc')
                    )
                ):
                    current = dict.fromkeys(COLUMNS)

                    current['source_file'] = source
                    current['utc_time'] = p[0]

                    current['_seen_rmc'] = False
                    current['_gsv'] = {}

                    rows.append(current)

                # RMC: 위치, 속도, 날짜
                if kind == 'RMC' and len(p) >= 9:

                    current['_seen_rmc'] = True

                    current.update(
                        rmc_status=p[1],
                        date_ddmmyy=p[8],
                        speed_kmh=(
                            round(float(p[6]) * 1.852, 4)
                            if p[6] else None
                        )
                    )

                    current['latitude_deg'] = coord(
                        p[2], p[3]
                    )

                    current['longitude_deg'] = coord(
                        p[4], p[5]
                    )

                # GGA: 위성 수, HDOP, 고도
                if kind == 'GGA' and len(p) >= 9:

                    current.update(
                        fix_quality=(
                            int(p[5]) if p[5] else None
                        ),
                        satellites_used=(
                            int(p[6]) if p[6] else None
                        ),
                        hdop=num(p[7]),
                        altitude_m=num(p[8])
                    )

                    if current['latitude_deg'] is None:
                        current['latitude_deg'] = coord(
                            p[1], p[2]
                        )

                    if current['longitude_deg'] is None:
                        current['longitude_deg'] = coord(
                            p[3], p[4]
                        )

            # GSV: 위성 신호 정보
            elif kind == 'GSV' and len(p) >= 3:

                total, index, visible = map(int, p[:3])

                if not (1 <= index <= total <= 20):
                    continue

                if index == 1:
                    gsv_pending[talker] = {
                        'total': total,
                        'next': 1,
                        'visible': visible,
                        'cn0': []
                    }

                state = gsv_pending.get(talker)

                if (
                    not state
                    or state['next'] != index
                    or state['total'] != total
                ):
                    continue

                # 위성별 C/N0 추출
                for j in range(3, len(p) - 3, 4):

                    strength = num(p[j + 3])

                    if strength is not None:
                        state['cn0'].append(strength)

                state['next'] += 1

                if index == total and current is not None:

                    current['_gsv'][talker] = {
                        'visible': visible,
                        'cn0': state['cn0'][:]
                    }

                    del gsv_pending[talker]

        except (ValueError, IndexError):
            stats['malformed_fields'] += 1

    # 6. 위성 신호 통계 계산
    for row in rows:

        groups = row.pop('_gsv')
        row.pop('_seen_rmc', None)

        row['gps_visible'] = (
            groups.get('GP', {}).get('visible')
        )

        row['glonass_visible'] = (
            groups.get('GL', {}).get('visible')
        )

        strengths = [
            value
            for group in groups.values()
            for value in group['cn0']
        ]

        # 신호 세기 측정 수
        row['cn0_count'] = len(strengths)

        # 평균 신호 세기
        row['cn0_mean_dbhz'] = (
            round(sum(strengths) / len(strengths), 3)
            if strengths else None
        )

        # 최대 신호 세기
        row['cn0_max_dbhz'] = (
            max(strengths)
            if strengths else None
        )

        # 추가: 위성 간 C/N0 표준편차
        row['cn0_xsat_std_db'] = (
            round(statistics.pstdev(strengths), 3)
            if len(strengths) >= 2
            else None
        )

    stats['epochs'] = len(rows)

    return rows, stats


# 7. 원본 파일 읽기
def sources(path):

    if path.suffix.lower() == '.zip':

        with zipfile.ZipFile(path) as archive:

            for name in sorted(archive.namelist()):

                if name.lower().endswith('.nmea'):
                    yield name, archive.read(name)

    elif path.is_dir():

        for file in sorted(path.rglob('*.nmea')):

            yield (
                str(file.relative_to(path)),
                file.read_bytes()
            )

    else:
        yield path.name, path.read_bytes()


# 8. 메인 실행 함수
def main():

    ap = argparse.ArgumentParser(
        description='GNSS NMEA GGA/RMC/GSV -> CSV'
    )

    ap.add_argument(
        'input',
        type=Path,
        help='raw.nmea, 폴더 또는 data.zip'
    )

    ap.add_argument(
        '-o',
        '--output',
        type=Path,
        default=Path('gnss_parsed.csv')
    )

    args = ap.parse_args()

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    total = Counter()
    count = 0

    with args.output.open(
        'w',
        newline='',
        encoding='utf-8-sig'
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=COLUMNS
        )

        writer.writeheader()

        for name, data in sources(args.input):

            rows, stats = parse_bytes(data, name)

            writer.writerows(rows)

            total.update(stats)
            count += 1

            print(
                f'{name}: {len(rows)} epochs, '
                f'{stats["checksum_failed"]} checksum failures'
            )

    print(
        f'\n완료: {count}개 파일, '
        f'{total["epochs"]}행 -> {args.output}'
    )

    print('원본 NMEA 파일은 수정하지 않았습니다.')
    print('위성 간 C/N0 표준편차 컬럼이 추가되었습니다.')
    print('주의: GSV는 앞선 RMC/GGA 측정 시점에 연결됩니다.')


if __name__ == '__main__':
    main()
