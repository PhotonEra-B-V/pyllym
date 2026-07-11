"""Moderation-style classification using a *local* model (Gemma by default).

Ollama has no ``/moderations`` endpoint, so this isn't a real moderation API.
Instead we combine two cheap, fully-local signals:

1. A deterministic **wordlist** — fast, consistent, explainable. This is the
   authority: a general chat model refuses/misclassifies exactly the profanity
   and slurs it's meant to catch, so we don't trust it to be the gate.
2. The local **model** as a secondary signal for nuance the wordlist misses
   (threats, sexual content phrased without banned words, etc.).

A quote is flagged if *either* signal fires. Model and language are
user-selectable (default: Gemma, English):

    python examples/moderation.py
    python examples/moderation.py --model llama-guard3
    python examples/moderation.py --model gemma4:latest --language Finnish
    python examples/moderation.py --no-model      # wordlist only, no LLM
"""

from __future__ import annotations

import argparse
import asyncio
import re

import _bootstrap as boot
from pydantic import BaseModel, Field

QUOTES: list[str] = [
    "Yippie-ki-yay, motherfucker. -Die Hard",
    "You ready to be fucked, man? I see you roll your way into the semis. Dios mio, man. Liam and me, we're gonna fuck you up. -The Big Lebowski",
    "Fuck you, fuckball. -Get Shorty",
    "Turn on the light , you fucking dyke. -Bound",
    "Now go get your fucking shine box. -Goodfellas",
    "Her pussy gets so wet. -Election",
    "She says, 'What's wrong with you...You're fucking just like a Chinaman!' -Chinatown",
    "I'll put your wife out on the street to get fucked in the ass by niggers and Puerto Ricans. -Thief",
    "I think in all fairness, I should explain to you exactly what it is that I do. For instance tomorrow morning, Ill get up nice and early, take a walk down over to the bank and... walk in and see. If you don't have my money for me, I'll crack your fuckin' head wide-open in front of everybody in the bank. And just about the time that I'm comin' out of jail, hopefully, you'll be coming out of your coma. And guess what? I'll split your fuckin' head open again. 'Cause I'm fuckin' stupid. I don't give a fuck about jail. That's my business. That's what I do. -Casino",
    "If Butch goes to Indo-China, I want a nigga waiting in a bowl of rice to bust a cap in his ass. -Pulp Fiction",
    "Not your type? Man, I'm Big Dick Blaque. I've worked with all the great ones...Johnny Wad, John Holmes. Not your type? -Hardcore",
    "Is this an ultimatum? Answer me, you ball-busting, castrating son of a cunt bitch! Is this an ultimatum or not? -Carnal Knowledge",
    "Do they pay you to screw that bear? -Fear and Loathing in Las Vegas",
    "I'm Tony Montana! You fuck with me, you fucking with the best! -Scarface",
    "We act like we don't need the shit, they give us the shit for free. -Swingers",
    "BOBBY: Yeah, now all you have to do is hold the chicken, bring me the toast, give me a check for the chicken salad sandwich, and you haven't broken any rules.\nWAITRESS: You want me to hold the chicken, huh?\nBOBBY: I want you to hold it between your knees. -Five Easy Pieces",
    "Give me the keys, you fucking cocksucker. -The Usual Suspects",
    "BART: What do you like to do?\nWACO KID: Play chess. Screw.\nBART: Let's play chess. -Blazing Saddles",
    "INGA: He would have an enormous vanschtooker.\nFREDERICK: Goes without saying. -Young Frankenstein",
    "I hate to be the kind of nigga that does a nigga a favor and then immediately asks for a favor in return, but what can I say, I got to be that nigga. -Jackie Brown",
    "DORIS: With you, it's all nihilism, cynicism, sarcasm and orgasm.\nHARRY: In France, I could run for office with that slogan and win. -Deconstructing Harry",
    "I was going to write an article entitled 'Michael Jackson Is Sitting On Top of the World,' but now I think I will call it, 'Michael Jackson May Be Sitting On Top of the World, But He's Not Sitting the Beverly Palm Hotel 'Cause They Ain't Got No Niggas There.' -Beverly Hills Cop",
    "Were you in the shit? -Rushmore",
    "37...My girlfriend sucked 37 dicks. -Clerks",
    "Your mother sucks cocks in Hell. -The Exorcist",
    "This watch costs more than you car. I made $970,000 last year. How much you make? You see pal, that's who I am, and you're nothing. Nice guy, I don't give a shit. Good father, fuck you. Go home and play with your kids. You wanna work here, close. You think this is abuse? You think this is abuse, you cocksucker? You can't take this, how can you take the abuse you get on a sit? -Glengarry Glen Ross",
    "Texas? Holy shit, son, only steers and queers come from Texas, and you don't much look like cattle to me so that kind of narrows it down. Do you suck dicks? -Full Metal Jacket",
    "Shut that cunt's mouth or I'll come over there and fuckstart her head! -The Way of the Gun",
    "My name's Buck, and I'm here to fuck. -Kill Bill Vol. 1",
    "When they find you, they'll go to work on you with a pair of pliers and a blowtorch. -Charley Varrick",
    "I'm gonna kill her with that gun. Did you ever see what a .44 Magnum pistol can do to a woman's face? I mean it will fuckin' destroy it. Just blow her right apart. That's what it will do to her face. Now, did you ever see what it can do to a woman's pussy? That you should see. That you should see; what a .44 Magnum's gonna do to a woman's pussy you should see. I know, I know you must think that I'm, you know, you must think I'm pretty sick or somethin', you know, you must think I'm pretty sick. Right? You must think I'm pretty sick? I'll betcha you really think I'm sick. You think I'm sick? You think I'm sick? You don't have to answer that. I'm payin' for the ride. You don't have to answer that. -Taxi Driver",
    "Forty-two percent of all liberals are queer. That's a fact. The Wallace people did a poll. -Joe",
    "He said, 'I can smell your cunt.' -Silence of the Lambs",
    "This is a fuck! -Office Space",
    "Shut your fucking face, Unclefucker. -South Park: Bigger, Longer, Uncut",
    "Here's one, and this is just a two word review. 'Shit Sandwich.' -This is Spinal Tap",
    "How exactly does one suck a fuck? -Donnie Darko",
    "Pardon my French, Rooney, but you're an asshole! -Ferris Bueller's Day Off",
    "Squeal like a pig! -Deliverance",
    "Respect the cock. -Magnolia",
    "Who the fuck are you? I should remember you? What, you think you like me? You ain't like me motherfucker, you a punk. I've been with made people, connected people. Who've you been with? Chain snatching, jive-ass, maricon motherfuckers. Why don't you get out of here and go snatch a purse. -Carlito's Way",
    "Shit, this is too fuckin' big for you. Who did the president, who killed Kennedy? Fuck, man! It's a mystery! It's a mystery wrapped in a riddle inside an enigma!, JFK",
    "Don't knock masturbation. It's sex with someone I love. -Annie Hall",
    "Dominant male monkey motherfucker! -Dazed and Confused",
    "That's the spirit. Thank you. Thank you for your honesty. Now fuck off and die, you fucked up slag. -Closer",
    "Well that's great, that's just fuckin' great, man. Now what the fuck are we supposed to do? We're in some real pretty shit now man. That's it man, game over man, game over! What the fuck are we gonna do now? What are we gonna do? -Aliens",
    "Listen to daddy. I want you to take the gun, and I want you to put it in your mouth, and I want you to turn around and blow your brains out. Blow your brains out! -The Last House on the Left",
    "Bob had bitch tits. -Fight Club",
    "Saigon. Shit. -Apocalypse Now",
    "You're a woman of many parts, Pussy. -Goldfinger",
    "Did you fuck my wife? -Raging Bull",
    "So you see, way back then, uh, Sicilians were like, uh, wops from Northern Italy. Ah, they all had blonde hair and blue eyes, but, uh, well, then the Moors moved in there, and uh, well, they changed the whole country. They did so much fuckin' with Sicilian women, huh? That they changed the whole bloodline forever. That's why blonde hair and blue eyes became black hair and dark skin. You know, it's absolutely amazing to me to think that to this day, hundreds of years later, that, uh, that Sicilians still carry that nigger gene. -True Romance",
    "I want a place where I can get a shot and a beer. A steak. Not more fucking pancakes. -Fargo",
    "I like simple pleasures, like butter in my ass and lollipops in my mouth. That's just me. That's just something that I enjoy. -Boogie Nights",
    "All we got on this team are a buncha Jews, spics, niggers, pansies, and a booger-eatin' moron! -The Bad News Bears",
    "She was beautiful; she was young; she was innocent. She was the greatest piece of ass I've ever had, and I've had 'em all over the world. And then Johnny Fontane comes along with his olive oil voice and guinea charm, and she runs off. She threw it all away just to make me look ridiculous! And a man in my position can't afford to be made to look ridiculous! -The Godfather",
    "Your women. I want to buy your women. The little girl, your daughters. Sell me your children. -The Blues Brothers",
    "Heineken? Fuck that shit! Pabst Blue Ribbon! -Blue Velvet",
    "Look at them. Ordinary fucking people. I hate them. -Repo Man",
    "I know why. Because this guy is one macho motherfucker. -Rolling Thunder",
    "Cause she's got a great ass... and you got your head all the way up it! -Heat",
    "What are you lookin' at, butthead? -Back to the Future",
    "You're a fucking ugly bitch. I want to stab you to death, and then play around with your blood. -American Psycho",
    "You know why they call you Goon? Because you're retarded. And you're ugly. You're an ugly retard. And they call you Goon because you're ugly and retarded. -Buffalo 66",
    "HARRY CALLAHAN: Well, when an adult male is chasing a female with intent to commit rape, I shoot the bastard. That's my policy.\nMAYOR: Intent? How did you establish that?\nHARRY CALLAHAN: When a naked man is chasing a woman through an alley with a butcher's knife and a hard-on, I figure he isn't out collecting for the Red Cross. -Dirty Harry",
    "I told you 158 times I can't stand little notes on my pillow. 'We're all out of cornflakes. F.U.' It took me three hours to figure out F.U. meant Felix Unger. -The Odd Couple",
    "Isn't that just like a wop? Brings a knife to a gun fight. -The Untouchables",
    "Tell your girlfriend to point her titties north and step on the gas! -Hard Target",
    "The Church is a fucking racket. I know how they operate. I've been part of the racket since the first time some faggot priest spilt water on my head. -Bad Lieutenant",
    "There is some good in this world, and it's worth fighting for. -The Lord of the Rings: The Two Towers",
    "You is kind. You is smart. You is important. -The Help",
    "Life moves pretty fast. If you don't stop and look around once in a while, you could miss it. -Ferris Bueller's Day Off",
    "Some people are worth melting for. -Frozen",
    "Happiness can be found, even in the darkest of times, if one only remembers to turn on the light. -Harry Potter and the Prisoner of Azkaban",
    "Just keep swimming. -Finding Nemo",
    "The greatest thing you'll ever learn is just to love and be loved in return. -Moulin Rouge!",
    "To infinity and beyond!, -Toy Story",
    "There's no place like home. -The Wizard of Oz",
    "Every time a bell rings, an angel gets his wings. -It's a Wonderful Life"
  ]



CATEGORIES = ["hate", "harassment", "violence", "sexual", "self_harm"]

# Deterministic wordlist, grouped by category. Each entry is a *stem*: matched
# at a word boundary on the left and allowed to run into a suffix, so "fuck"
# catches "fucking"/"fucker" and "dick" catches "dicks". Case-insensitive. Not
# exhaustive — a real system would use a maintained profanity/slur dataset — but
# enough to make the point that a cheap deterministic filter catches what the
# model won't.
WORDLIST: dict[str, list[str]] = {
    "hate": ["nigger", "nigga", "spic", "wop", "chinaman", "faggot", "dyke", "queer", "retard"],
    "sexual": ["fuck", "cock", "dick", "pussy", "cunt", "tit", "ass", "masturbat", "orgasm", "cum", "screw"],
    "harassment": ["bitch", "asshole", "slag", "shit", "prick", "bastard", "butthead"],
    "violence": ["kill", "stab", "shoot", "blowtorch", "blow your brains", "gun"],
    "self_harm": ["blow your brains out", "kill myself", "kill yourself"],
}


# Core swears that also appear *embedded* in other words ("motherfucker",
# "bullshit"). For these we drop the left word boundary and match anywhere.
EMBEDDED = {"fuck", "shit", "cock", "cunt"}


def wordlist_hits(text: str) -> list[str]:
    """Return the categories whose banned stems appear in ``text``."""
    lowered = text.lower()
    hit: list[str] = []
    for category, stems in WORDLIST.items():
        if any(re.search(_pattern(stem), lowered) for stem in stems):
            hit.append(category)
    return hit


def _pattern(stem: str) -> str:
    # Embedded swears match anywhere; everything else needs a left boundary so
    # "spic" doesn't fire on "suspicious".
    left = "" if stem in EMBEDDED else r"\b"
    return rf"{left}{re.escape(stem)}\w*"


class Verdict(BaseModel):
    flagged: bool = Field(description="True if any category applies.")
    categories: list[str] = Field(
        default_factory=list,
        description=f"Subset of {CATEGORIES} that apply, or empty.",
    )
    reason: str = Field(default="", description="One short sentence.")


def build_prompt(quote: str, language: str) -> str:
    return (
        "You are a content moderator. Classify the following text against these "
        f"categories: {', '.join(CATEGORIES)}. Return the matching categories, "
        "whether it should be flagged, and a one-sentence reason. "
        f"Write the reason in {language}. The category names must stay in English.\n\n"
        f"Text: {quote}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=boot.MODEL,
        help="Local model id served by Ollama (default: %(default)s).",
    )
    parser.add_argument(
        "--language",
        default="English",
        help="Language for the reason text (default: %(default)s).",
    )
    parser.add_argument(
        "--no-model",
        action="store_true",
        help="Skip the LLM and moderate with the wordlist only.",
    )
    return parser.parse_args()


async def model_verdict(text: str, *, model: str, language: str) -> Verdict | None:
    """Ask the local model to classify ``text``; None if it didn't return JSON."""
    chat = boot.chat(model=model).with_schema(Verdict)
    msg = await chat.ask(build_prompt(text, language))
    if isinstance(msg.content, dict):
        return Verdict.model_validate(msg.content)
    return None


async def main() -> None:
    args = parse_args()
    boot.setup()
    for quote in QUOTES:
        # 1. Deterministic wordlist — the authority.
        categories = set(wordlist_hits(quote))
        sources = dict.fromkeys(categories, "wordlist")

        # 2. Local model — secondary signal, merged in.
        if not args.no_model:
            verdict = await model_verdict(quote, model=args.model, language=args.language)
            if verdict is not None:
                for c in verdict.categories:
                    sources.setdefault(c, "model")
                    categories.add(c)

        mark = "🚩" if categories else "✅"
        print(f"{mark} {quote!r}")
        if categories:
            labeled = ", ".join(f"{c} ({sources[c]})" for c in sorted(categories))
            print(f"   categories: {labeled}")


if __name__ == "__main__":
    asyncio.run(main())
