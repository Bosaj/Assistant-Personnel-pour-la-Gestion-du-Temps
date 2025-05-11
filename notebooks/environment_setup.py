import numpy as np
import pandas as pd
import random
import gym
from gym import spaces

class ActivityPatterns:
    """Classe pour gérer les modèles d'activités et préférences temporelles"""
    
    def __init__(self, data):
        self.data = data
        self._initialize_patterns()
        
    def _initialize_patterns(self):
        """Initialise les modèles d'activités à partir des données"""
        # Calculer les distributions horaires pour chaque activité
        self.activity_time_prefs = {}
        
        # Assurez-vous que 'hour' et 'ACTIVITY_NAME_ENC' existent dans les données
        if 'hour' in self.data.columns and 'ACTIVITY_NAME_ENC' in self.data.columns:
            # Pour chaque activité, calculer sa distribution par heure
            for activity in self.data['ACTIVITY_NAME_ENC'].unique():
                activity_data = self.data[self.data['ACTIVITY_NAME_ENC'] == activity]
                hour_dist = activity_data['hour'].value_counts(normalize=True).sort_index()
                
                # Remplir les heures manquantes avec des zéros
                all_hours = pd.Series(0.0, index=range(24))
                for hour, freq in hour_dist.items():
                    if 0 <= hour < 24:  # Vérifier que l'heure est valide
                        all_hours[hour] = freq
                
                self.activity_time_prefs[activity] = all_hours

    def get_time_preference_score(self, activity, hour):
        """Retourne un score de préférence pour l'activité à l'heure donnée"""
        if activity in self.activity_time_prefs and 0 <= hour < 24:
            return self.activity_time_prefs[activity][hour]
        return 0.01  # Score par défaut faible mais non nul

    def get_activity_sequence_score(self, activity1, activity2):
        """Évalue la pertinence de la séquence entre deux activités"""
        # Vous pourriez implémenter ici une logique pour évaluer
        # si activity2 suit bien activity1 (basé sur les données)
        return 0.5  # Valeur neutre par défaut


class ScheduleEnv(gym.Env):
    """
    Environnement pour l'optimisation de planning avec apprentissage par renforcement
    """
    
    def __init__(self, data, num_slots=24):
        super(ScheduleEnv, self).__init__()
        
        self.data = data
        self.num_slots = num_slots
        
        # Sauvegarder les activités uniques
        self.unique_activities = sorted(data['ACTIVITY_NAME_ENC'].unique())
        self.num_activities = len(self.unique_activities)
        
        # Dictionnaire pour mapper les codes aux noms d'activités
        # Cette partie est facultative et dépend de la présence de 'ACTIVITY_NAME' dans vos données
        if 'ACTIVITY_NAME' in data.columns:
            self.activity_names = {}
            for act_code in self.unique_activities:
                act_name = data[data['ACTIVITY_NAME_ENC'] == act_code]['ACTIVITY_NAME'].iloc[0]
                self.activity_names[act_code] = act_name
        else:
            # Si pas de noms d'activités, utiliser les codes comme noms
            self.activity_names = {act: f"Activity {act}" for act in self.unique_activities}
        
        # Action : choisir une activité pour un slot
        self.action_space = spaces.Discrete(self.num_activities)
        
        # Observation : 
        # - Slot horaire actuel (24 possibilités)
        # - Jour de la semaine (7 possibilités)
        # - Dernière activité choisie (num_activities possibilités)
        # - Indicateur d'activités déjà planifiées (vecteur binaire)
        obs_dim = 3 + self.num_slots  # slot actuel + jour + dernière action + activités planifiées
        self.observation_space = spaces.Box(low=0, high=1, shape=(obs_dim,), dtype=np.float32)
        
        # Analyser les modèles d'activités
        self.activity_patterns = ActivityPatterns(data)
        
        # Initialiser l'état
        self.reset()

    def reset(self):
        """Réinitialise l'environnement à un état initial"""
        self.current_slot = 0
        self.schedule = np.zeros(self.num_slots, dtype=np.int32)
        self.day_of_week = random.randint(0, 6)
        self.last_activity = 0  # Aucune activité précédente
        
        return self._get_observation()

    def _get_observation(self):
        """Retourne l'observation actuelle"""
        # Normaliser le slot pour qu'il soit entre 0 et 1
        slot_normalized = self.current_slot / self.num_slots
        
        # Normaliser le jour de la semaine
        day_normalized = self.day_of_week / 6
        
        # Normaliser la dernière activité
        last_act_normalized = self.last_activity / (self.num_activities - 1) if self.num_activities > 1 else 0
        
        # Créer un vecteur binaire indiquant les activités déjà planifiées
        planned_activities = np.zeros(self.num_slots)
        for i in range(self.current_slot):
            planned_activities[i] = 1
        
        # Combiner les composants pour former l'observation complète
        observation = np.concatenate([[slot_normalized, day_normalized, last_act_normalized], planned_activities])
        
        return observation

    def step(self, action):
        """
        Exécuter une action (choisir une activité pour le slot courant)
        """
        # Vérifier la validité de l'action
        if not self.action_space.contains(action):
            action = random.randint(0, self.num_activities - 1)
        
        # Stocker l'activité dans le planning
        self.schedule[self.current_slot] = action
        
        # Calculer la récompense
        reward = self._calculate_reward(action, self.current_slot)
        
        # Mettre à jour l'état
        self.last_activity = action
        self.current_slot += 1
        
        # Vérifier si l'épisode est terminé
        done = self.current_slot >= self.num_slots
        
        # Obtenir la nouvelle observation
        observation = self._get_observation()
        
        # Informations supplémentaires (pour débogage)
        info = {
            'current_slot': self.current_slot,
            'day_of_week': self.day_of_week,
            'activity': action
        }
        
        return observation, reward, done, info

    def _calculate_reward(self, action, time_slot):
        """Calcule la récompense pour l'action choisie"""
        reward = 0.0
        
        # Récompense basée sur les préférences horaires
        time_pref_score = self.activity_patterns.get_time_preference_score(action, time_slot)
        reward += time_pref_score * 2.0
        
        # Récompense basée sur la séquence d'activités
        if time_slot > 0:
            previous_action = self.schedule[time_slot - 1]
            sequence_score = self.activity_patterns.get_activity_sequence_score(previous_action, action)
            reward += sequence_score
        
        # Pour éviter des récompenses trop faibles
        reward = max(reward, 0.1)
        
        return reward

    def get_activity_name(self, activity_code):
        """Retourne le nom de l'activité correspondant au code"""
        return self.activity_names.get(activity_code, f"Activity {activity_code}")

    def render(self, mode='human'):
        """Affiche l'état actuel du planning"""
        if mode == 'human':
            print(f"🕒 Slot actuel: {self.current_slot}/{self.num_slots}")
            print(f"📅 Jour: {['Dim', 'Lun', 'Mar', 'Mer', 'Jeu', 'Ven', 'Sam'][self.day_of_week]}")
            print("📋 Planning actuel:")
            
            for i, act in enumerate(self.schedule):
                if i < self.current_slot:
                    print(f"  {i:02d}:00 - {self.get_activity_name(act)}")
                else:
                    print(f"  {i:02d}:00 - [Non planifié]")
            
            print("")
        
        return None