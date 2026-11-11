import json
import os
from datetime import datetime
from pathlib import Path
import re

def analyze_attack_scenarios(base_path):
    total_attack_time = 0  # in seconds
    scenarios_data = []
    earliest_time = None
    latest_time = None
    errors = []
    
    # Walk through all directories to find scenario.json files
    for root, dirs, files in os.walk(base_path):
        if 'scenario.json' in files:
            scenario_file = os.path.join(root, 'scenario.json')
            
            try:
                with open(scenario_file, 'r') as f:
                    data = json.load(f)
                
                scenario_id = data.get('scenario_id', 'Unknown')
                attack_type = data.get('attack_type', 'Unknown')
                
                # Determine if dynamic or stationary from file path
                is_dynamic = 'dynamic' in scenario_file
                movement_type = 'dynamic' if is_dynamic else 'stationary'
                
                # Extract attack log timestamps
                attack_log = data.get('attack_log', [])
                print(f"\n🔍 Checking scenario {scenario_id} ({attack_type}) - {movement_type.upper()}:")
                print(f"   Attack log entries: {len(attack_log)}")
                
                if attack_log:
                    for i, entry in enumerate(attack_log):
                        print(f"   Entry {i+1}: {entry}")
                
                if len(attack_log) >= 2:
                    # Group start/end events by attack ID or sequence
                    attack_sessions = {}
                    scenario_total_time = 0
                    
                    for entry in attack_log:
                        event = entry.get('event', '').lower()
                        timestamp = entry.get('timestamp_utc', '')
                        
                        # Extract attack ID from event (e.g., "2.1.3" from "Jamming started (2.1.3)")
                        attack_id_match = re.search(r'\(([^)]+)\)', entry.get('event', ''))
                        attack_id = attack_id_match.group(1) if attack_id_match else 'default'
                        
                        if attack_id not in attack_sessions:
                            attack_sessions[attack_id] = {'start': None, 'end': None}
                        
                        if 'start' in event or 'began' in event:
                            attack_sessions[attack_id]['start'] = timestamp
                        elif 'end' in event or 'stop' in event or 'finish' in event:
                            attack_sessions[attack_id]['end'] = timestamp
                    
                    print(f"   Found {len(attack_sessions)} attack sessions:")
                    
                    # Calculate duration for each attack session
                    for attack_id, session in attack_sessions.items():
                        if session['start'] and session['end']:
                            start_time = datetime.fromisoformat(session['start'].replace('Z', '+00:00'))
                            end_time = datetime.fromisoformat(session['end'].replace('Z', '+00:00'))
                            
                            duration = (end_time - start_time).total_seconds()
                            scenario_total_time += duration
                            
                            print(f"     • Attack {attack_id}: {duration/60:.1f} minutes")
                            
                            # Track overall timespan
                            if earliest_time is None or start_time < earliest_time:
                                earliest_time = start_time
                            if latest_time is None or end_time > latest_time:
                                latest_time = end_time
                        else:
                            print(f"     • Attack {attack_id}: Missing start or end timestamp")
                    
                    if scenario_total_time > 0:
                        total_attack_time += scenario_total_time
                        print(f"   ✅ Total scenario duration: {scenario_total_time/60:.1f} minutes")
                        
                        scenarios_data.append({
                            'scenario_id': scenario_id,
                            'attack_type': attack_type,
                            'movement_type': movement_type,
                            'duration_minutes': scenario_total_time / 60,
                            'attack_sessions': len(attack_sessions),
                            'date': data.get('date')
                        })
                    else:
                        error_msg = f"No valid attack sessions found in {scenario_id}"
                        print(f"   ❌ {error_msg}")
                        errors.append(error_msg)
                else:
                    error_msg = f"Insufficient attack log entries in {scenario_id}: {len(attack_log)}"
                    print(f"   ❌ {error_msg}")
                    errors.append(error_msg)
                        
            except Exception as e:
                error_msg = f"Error reading {scenario_file}: {e}"
                print(f"❌ {error_msg}")
                errors.append(error_msg)
    
    # Calculate results
    total_dataset_time = (latest_time - earliest_time).total_seconds() if earliest_time and latest_time else 0
    
    print(f"\n📊 **ATTACK SCENARIO TIME ANALYSIS**")
    print(f"="*60)
    print(f"🔢 **Total Scenarios Found**: {len(scenarios_data)}")
    print(f"⏱️  **Total Attack Time**: {total_attack_time/3600:.2f} hours ({total_attack_time/60:.1f} minutes)")
    print(f"📅 **Dataset Timespan**: {total_dataset_time/3600:.2f} hours ({total_dataset_time/86400:.1f} days)")
    print(f"🚀 **Earliest Attack**: {earliest_time}")
    print(f"🏁 **Latest Attack**: {latest_time}")
    
    if errors:
        print(f"\n⚠️  **ERRORS ENCOUNTERED**: {len(errors)}")
        for error in errors[:5]:
            print(f"   • {error}")
        if len(errors) > 5:
            print(f"   ... and {len(errors)-5} more errors")
    
    # Movement type breakdown (Dynamic vs Stationary)
    movement_stats = {}
    for scenario in scenarios_data:
        movement_type = scenario['movement_type']
        if movement_type not in movement_stats:
            movement_stats[movement_type] = {'count': 0, 'total_time': 0, 'sessions': 0}
        movement_stats[movement_type]['count'] += 1
        movement_stats[movement_type]['total_time'] += scenario['duration_minutes']
        movement_stats[movement_type]['sessions'] += scenario['attack_sessions']
    
    print(f"\n🏃 **MOVEMENT TYPE BREAKDOWN**:")
    print(f"="*60)
    total_time_minutes = total_attack_time / 60
    for movement_type, stats in movement_stats.items():
        percentage = (stats['total_time'] / total_time_minutes) * 100 if total_time_minutes > 0 else 0
        print(f"  📍 **{movement_type.upper()}**:")
        print(f"     • Scenarios: {stats['count']}")
        print(f"     • Attack Sessions: {stats['sessions']}")
        print(f"     • Total Time: {stats['total_time']:.1f} minutes ({stats['total_time']/60:.2f} hours)")
        print(f"     • Percentage: {percentage:.1f}% of total attack time")
        print(f"     • Avg per scenario: {stats['total_time']/stats['count']:.1f} minutes")
    
    # Attack type breakdown
    attack_types = {}
    total_sessions = 0
    for scenario in scenarios_data:
        attack_type = scenario['attack_type']
        if attack_type not in attack_types:
            attack_types[attack_type] = {
                'count': 0, 'total_time': 0, 'sessions': 0,
                'dynamic': {'count': 0, 'time': 0, 'sessions': 0},
                'stationary': {'count': 0, 'time': 0, 'sessions': 0}
            }
        
        attack_types[attack_type]['count'] += 1
        attack_types[attack_type]['total_time'] += scenario['duration_minutes']
        attack_types[attack_type]['sessions'] += scenario['attack_sessions']
        
        # Add to movement type subcategory
        movement = scenario['movement_type']
        attack_types[attack_type][movement]['count'] += 1
        attack_types[attack_type][movement]['time'] += scenario['duration_minutes']
        attack_types[attack_type][movement]['sessions'] += scenario['attack_sessions']
        
        total_sessions += scenario['attack_sessions']
    
    print(f"\n🎯 **ATTACK TYPE BREAKDOWN**:")
    print(f"="*60)
    for attack_type, stats in attack_types.items():
        percentage = (stats['total_time'] / total_time_minutes) * 100 if total_time_minutes > 0 else 0
        print(f"  ⚡ **{attack_type.upper()}**:")
        print(f"     • Total: {stats['count']} scenarios, {stats['sessions']} sessions, {stats['total_time']:.1f} minutes ({percentage:.1f}%)")
        
        # Dynamic breakdown
        if stats['dynamic']['count'] > 0:
            dyn_pct = (stats['dynamic']['time'] / stats['total_time']) * 100 if stats['total_time'] > 0 else 0
            print(f"       - Dynamic: {stats['dynamic']['count']} scenarios, {stats['dynamic']['sessions']} sessions, {stats['dynamic']['time']:.1f} min ({dyn_pct:.1f}%)")
        
        # Stationary breakdown
        if stats['stationary']['count'] > 0:
            stat_pct = (stats['stationary']['time'] / stats['total_time']) * 100 if stats['total_time'] > 0 else 0
            print(f"       - Stationary: {stats['stationary']['count']} scenarios, {stats['stationary']['sessions']} sessions, {stats['stationary']['time']:.1f} min ({stat_pct:.1f}%)")
    
    print(f"\n📈 **FINAL SUMMARY**:")
    print(f"="*60)
    print(f"  • Total Scenarios: {len(scenarios_data)}")
    print(f"  • Total Attack Sessions: {total_sessions}")
    print(f"  • Total Attack Time: {total_time_minutes:.1f} minutes ({total_time_minutes/60:.2f} hours)")
    print(f"  • Average Duration per Session: {total_time_minutes/total_sessions:.1f} minutes")
    print(f"  • Average Duration per Scenario: {total_time_minutes/len(scenarios_data):.1f} minutes")
    
    # Movement type ratios
    if len(movement_stats) >= 2:
        dynamic_time = movement_stats.get('dynamic', {}).get('total_time', 0)
        stationary_time = movement_stats.get('stationary', {}).get('total_time', 0)
        print(f"\n  🔄 **DYNAMIC vs STATIONARY RATIO**:")
        print(f"     • Dynamic: {dynamic_time:.1f} min ({(dynamic_time/total_time_minutes)*100:.1f}%)")
        print(f"     • Stationary: {stationary_time:.1f} min ({(stationary_time/total_time_minutes)*100:.1f}%)")
        
        if stationary_time > 0:
            ratio = dynamic_time / stationary_time
            print(f"     • Ratio (Dynamic:Stationary): {ratio:.2f}:1")
    
    return scenarios_data, total_attack_time, total_dataset_time

# Run the analysis
base_path = "/home/issam/Second_year/attack_scenarios_tree"
scenarios, total_attacks, total_dataset = analyze_attack_scenarios(base_path)