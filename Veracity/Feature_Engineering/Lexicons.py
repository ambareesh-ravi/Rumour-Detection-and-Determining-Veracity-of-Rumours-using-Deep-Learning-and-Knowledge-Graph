

false_synonyms = ['false', 'bogus', 'deceitful', 'dishonest',
                          'distorted', 'erroneous', 'fake', 'fanciful',
                          'faulty', 'fictitious', 'fraudulent',
                          'improper', 'inaccurate', 'incorrect',
                          'invalid', 'misleading', 'mistaken', 'phony',
                          'specious', 'spurious', 'unfounded', 'unreal',
                          'untrue',  'untruthful', 'apocryphal',
                          'beguiling', 'casuistic', 'concocted',
                          'cooked-up', 'counterfactual',
                          'deceiving', 'delusive', 'ersatz',
                          'fallacious', 'fishy', 'illusive', 'imaginary',
                          'inexact', 'lying', 'mendacious',
                          'misrepresentative', 'off the mark', 'sham',
                          'sophistical', 'trumped up', 'unsound']
						  
false_antonyms = ['accurate', 'authentic', 'correct', 'fair',
                       'faithful', 'frank', 'genuine', 'honest', 'moral',
                          'open', 'proven', 'real', 'right', 'sincere',
                          'sound', 'true', 'trustworthy', 'truthful',
                          'valid', 'actual', 'factual', 'just', 'known',
                          'precise', 'reliable', 'straight', 'substantiated']

negation_words = ['not', 'no', 'nobody', 'nothing', 'none', 'never',
                         'neither', 'nor', 'nowhere', 'hardly', 'scarcely',
                         'barely', 'don', 'isn', 'wasn', 'shouldn', 'wouldn',
                         'couldn', 'doesn']

SpeechAct = {}
SpeechAct['SpeechAct_ORDER'] = ['command', 'demand', 'tell', 'direct','instruct', 'require', 'prescribe','order']
SpeechAct['SpeechAct_ASK1'] = ['ask', 'request', 'beg', 'bespeech','implore','appeal', 'plead', 'intercede',
								'apply','urge', 'persuade', 'dissuade', 'convince']
SpeechAct['SpeechAct_ASK2'] = ['ask', 'inquire', 'enquire', 'interrogate','question', 'query']
SpeechAct['SpeechAct_CALL'] = ['call', 'summon', 'invite', 'call on','call for', 'order', 'book', 'reserve']
SpeechAct['SpeechAct_FORBID'] = ['forbid', 'prohibit', 'veto', 'refuse','decline', 'reject', 'rebuff', 'renounce',
                                     'cancel', 'resign', 'dismiss']
SpeechAct['SpeechAct_PERMIT'] = ['permit', 'allow', 'consent', 'accept','agree', 'approve', 'disapprove','authorize', 'appoint']
SpeechAct['SpeechAct_ARGUE'] = ['argue', 'disagree', 'refute', 'contradict','counter', 'deny', 'recant', 'retort','quarrel']
SpeechAct['SpeechAct_REPRIMAND'] = ['reprimand', 'rebuke', 'reprove','admonish', 'reproach', 'nag','scold', 'abuse', 'insult']
SpeechAct['SpeechAct_MOCK'] = ['ridicule', 'joke']
SpeechAct['SpeechAct_BLAME'] = ['blame', 'criticize', 'condemn','denounce', 'deplore', 'curse']
SpeechAct['SpeechAct_ACCUSE'] = ['accuse', 'charge', 'challenge', 'defy',
                                     'dare']
SpeechAct['SpeechAct_ATTACK'] = ['attack', 'defend']
SpeechAct['SpeechAct_WARN '] = ['warn', 'threaten', 'blackmail']
SpeechAct['SpeechAct_ADVISE '] = ['advise', 'councel', 'consult','recommend', 'suggest', 'propose','advocate']
SpeechAct['SpeechAct_OFFER '] = ['offer', 'volunteer', 'grant', 'give']
SpeechAct['SpeechAct_PRAISE '] = ['praise', 'commend', 'compliment','boast', 'credit']
SpeechAct['SpeechAct_PROMISE '] = ['promise', 'pledge', 'vow', 'swear','vouch for', 'guarante']
SpeechAct['SpeechAct_THANK '] = ['thank', 'apologise', 'greet', 'welcome','farewell', 'goodbye',
									'introduce','bless', 'wish', 'congratulate']
SpeechAct['SpeechAct_FORGIVE '] = ['forgive', 'excuse', 'justify','absolve', 'pardon', 'convict','acquit', 'sentence']
SpeechAct['SpeechAct_COMPLAIN'] = ['complain', 'protest', 'object','moan', 'bemoan', 'lament', 'bewail']
SpeechAct['SpeechAct_EXCLAIM'] = ['exclaim', 'enthuse', 'exult', 'swear','blaspheme']
SpeechAct['SpeechAct_GUESS'] = ['guess', 'bet', 'presume', 'suspect','suppose', 'wonder', 
								'speculate','conjecture', 'predict', 'forecast','prophesy']
SpeechAct['SpeechAct_HINT'] = ['hint', 'imply', 'insinuate']
SpeechAct['SpeechAct_CONCLUDE'] = ['conclude', 'deduce', 'infer', 'gather','reckon',
									'estimate', 'calculate','count', 'prove', 'compare']
SpeechAct['SpeechAct_TELL'] = ['tell', 'report', 'narrate', 'relate','recount', 'describe', 'explain', 'lecture']
SpeechAct['SpeechAct_INFORM'] = ['inform', 'notify', 'announce', 'inform on', 'reveal']
SpeechAct['SpeechAct_SUMUP'] = ['sum up', 'summarize', 'recapitulate']
SpeechAct['SpeechAct_ADMIT'] = ['admit', 'acknowledge', 'concede','confess', 'confide']
SpeechAct['SpeechAct_ASSERT'] = ['assert', 'affirm', 'claim', 'maintain','contend', 'state', 'testify']
SpeechAct['SpeechAct_CONFIRM'] = ['confirm', 'assure', 'reassure']
SpeechAct['SpeechAct_STRESS'] = ['stress', 'emphasize', 'insist', 'repeat','point out', 'note', 'remind', 'add']
SpeechAct['SpeechAct_DECLARE'] = ['declare', 'pronounce', 'proclaim','decree', 'profess', 'vote', 'resolve','decide']
SpeechAct['SpeechAct_BAPTIZE'] = ['baptize', 'chirsten', 'name','excommunicate']
SpeechAct['SpeechAct_REMARK'] = ['remark', 'comment', 'observe']
SpeechAct['SpeechAct_ANSWER'] = ['answer', 'reply']
SpeechAct['SpeechAct_DISCUSS'] = ['discuss', 'debate', 'negotiate','bargain']
SpeechAct['SpeechAct_TALK'] = ['talk', 'converse', 'chat', 'gossip']