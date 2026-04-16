# Mudra Taxonomy — Hasta Reference Guide

Mudras (hand gestures) are the core visual language of Indian classical dance.  
NatyaVeda classifies **28 Asamyuta Hastas** (single-hand gestures) per frame.

---

## Asamyuta Hastas (Single-Hand, 28 Gestures)

These are the primary gestures encoded in the `mudra_head` output of DanceFormer.

| ID | Name | Fingers | Primary Meanings |
|----|------|---------|-----------------|
| 0  | Pataka | All 4 fingers + thumb bent | Cloud, forest, night, stopping |
| 1  | Tripataka | Ring finger bent | Crown, tree, sword, number 3 |
| 2  | Ardhapataka | Index + middle together | Knife, two, banana |
| 3  | Kartarimukha | Index + ring extended, middle bent | Quarrel, deer's eye, separation |
| 4  | Mayura | Thumb + index pinch | Peacock, cheek, rain |
| 5  | Ardhachandra | All fingers spread, thumb out | Moon, waist, spear |
| 6  | Arala | Index bent backward | Saraswati, drinking nectar |
| 7  | Shukatunda | Index curved forward | Arrow, shooting |
| 8  | Mushti | All fingers in fist | Wrestling, strength, holding |
| 9  | Shikhara | Thumb up from fist | Bow, pillar, husband, Shiva |
| 10 | Kapittha | Thumb + middle touch | Lakshmi, holding, milk |
| 11 | Katakamukha | Index + middle bent outward | Plucking flowers, garland |
| 12 | Suchi | Index pointed up | One, numeral, addressing crowd |
| 13 | Chandrakala | Thumb + index crescent | Moon on Shiva's head |
| 14 | Padmakosha | All fingers curved downward | Lotus, ball, round fruit |
| 15 | Sarpashirsha | All fingers together bent at middle | Snake, wave, applying sandal |
| 16 | Mrigashirsha | Pinky + ring bent, others extended | Deer, woman, fear |
| 17 | Simhamukha | Thumb + index + pinky extended | Lion, owl |
| 18 | Kangula | Thumb + ring touch | Fruit, bell |
| 19 | Alapadma | All 5 fingers wide spread | Full lotus, beauty, village |
| 20 | Chatura | 4 fingers close, thumb tucked | Clever, musk, gold, wind |
| 21 | Bhramara | Index on thumb, middle curved | Bee, parrot |
| 22 | Hamsasya | All 5 fingertips together | Swan, needle, thread, offer |
| 23 | Hamsapaksha | 4 fingers together, thumb apart | Count 6, offer water |
| 24 | Sandamsha | Index + middle pinching | Tongs, holding flower |
| 25 | Mukula | All fingertips together pointing up | Bud, eating, flower offering |
| 26 | Tamrachuda | Index curves down from pataka | Cock, river flow |
| 27 | Trishula | Index + middle + ring spread | Shiva's trident, three |

---

## Samyuta Hastas (Two-Hand, 24 Gestures)

Used for objects, concepts, and blessings requiring both hands.

| Name | Description |
|------|-------------|
| Anjali | Joined palms — salutation, greeting Namaskar |
| Kapota | Joined backs of hands — modesty, formal agreement |
| Karkata | Interlocked fingers — stiff, forest, rough texture |
| Swastika | Crossed wrists — crossroads, forest, auspicious symbol |
| Dola | Both arms hanging loose — neutral, natural walk |
| Pushpaputa | Cupped hands — collecting flowers, receiving blessings |
| Utsanga | Arms crossed at chest — embrace, wearing garland |
| Shivalinga | Right Shikhara on left Ardhachandra — Shiva worship |
| Katakavardhana | Katakamukha overlapping — coronation, prosperity |
| Kartariswastika | Crossed Kartarimukha — birds in sky, fighting |
| Shakata | Shikhara + Bhramara — demon, buffalo, Rakshasa |
| Shankha | Right Shikhara in left grasp — conch shell blowing |
| Chakra | Ardhachandra rotating — discus, wheel, Vishnu's Sudarshana |
| Samputa | Cupped Pataka — box, secret, guarding |
| Pasha | Linked Suchi fingers — bond, rope, capture |
| Kilaka | Linked Suchi fingers side by side — friendship, link |
| Matsya | Pataka on back of other Pataka — fish swimming |
| Kurma | Crossed fingers over — tortoise shell |
| Varaha | Right Shikhara on left Mrigashirsha — Vishnu's boar avatar |
| Garuda | Thumbs interlocked — Garuda eagle, Vishnu's vehicle |
| Nagabandha | Arms coiled like snake — serpent coil, Naga |
| Bherunda | Two Sarpashirsha facing outward — pair of birds |
| Avahittha | Pataka bending at wrist — concealment, shy gesture |
| Vajra | Shikhara in both hands crossing — Indra's thunderbolt |

---

## Finger Joint Labeling (MediaPipe 21-point)

```
        TIP (4)
         |
        DIP (3)
         |
        PIP (2)
         |
        MCP (1)
         |
   ── WRIST (0) ──
```

For each of 5 fingers × 4 joints = 20 joints + 1 wrist = **21 landmarks per hand**.

| Joint | Type | Abbreviation |
|-------|------|-------------|
| 0     | Wrist | WRI |
| 1,5,9,13,17 | Metacarpophalangeal | MCP |
| 2,6,10,14,18 | Proximal interphalangeal | PIP |
| 3,7,11,15,19 | Distal interphalangeal | DIP |
| 4,8,12,16,20 | Fingertip | TIP |

---

## Mudra Detection Pipeline

```
Frame input
    │
    ▼
MediaPipe Hands → 21 pts × 2 hands (x, y, z)
    │
    ├── RTMW hand kpts [91:133] → confidence-weighted fusion
    │
    ▼
HandFrame.mudra_features() → [139-dim vector]
    │   ├── raw landmarks: 42 pts × 3 = 126
    │   ├── right finger flexion angles: 5
    │   ├── left finger flexion angles:  5
    │   └── relative hand positions:     3
    │
    ▼
MudraRecognizer (geometric rules → or learned MLP head)
    │
    ▼
Mudra label + confidence per frame
```

---

## References

- **Natyashastra** (Bharata Muni, ~200 BCE–200 CE) — original codification
- **Abhinaya Darpana** (Nandikesvara) — practical performer's guide
- **Hastalakshanadipika** (Kerala tradition) — basis for Kathakali and Mohiniyattam hastas
