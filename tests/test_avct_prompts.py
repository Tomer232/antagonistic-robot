import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from roastcrowd.config.settings import AvctConfig
from roastcrowd.conversation.avct_manager import AvctManager

def test_avct_safety():
    cfg = AvctConfig(prompts_dir="../prompts")
    mgr = AvctManager(cfg)
    
    prompt = mgr.get_system_prompt("session_123", polar_level=3, category="G", subtype=3, modifiers=["M4"])
    
    assert "MANDATORY SAFETY BOUNDARIES" in prompt, "Missing safety boundaries!"
    assert "intensity level 3" in prompt, "Missing intensity!"
    assert "modifier M4" in prompt, "Missing modifiers!"
    assert "category G3" in prompt, "Missing category context!"
    
    print("test_avct_safety passed!")

def test_avct_risk():
    cfg = AvctConfig(prompts_dir="../prompts")
    mgr = AvctManager(cfg)
    
    assert mgr.get_risk_rating(3, "B", 1, []) == "Red", "Risk rating failed for 3B1"
    assert mgr.get_risk_rating(2, "G", 1, []) == "Red", "Risk rating failed for 2G1"
    assert mgr.get_risk_rating(2, "C", 1, []) == "Amber", "Risk rating failed for 2C1"
    assert mgr.get_risk_rating(0, "D", 1, []) == "Green", "Risk rating failed for 0D1"

    print("test_avct_risk passed!")

if __name__ == "__main__":
    test_avct_safety()
    test_avct_risk()
    print("All AVCT tests passed!")
