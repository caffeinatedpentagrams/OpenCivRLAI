# https://upload.wikimedia.org/wikipedia/commons/4/4c/Freeciv-2.1.8_technology_tree.png

class Technology:
    """Base class for technologies"""
    def __init__(self, name, requirements, cost):
        self.name = name
        self.requirements = requirements
        self.researched = False
        self.cost = cost  # TODO check if this wasn't turns, but I think it is science cost
        self.progress = 0

    def __str__(self):
        return self.name


class TechnologyTree:
    """Technology tree"""
    def __init__(self):  # TODO Correct costs!
        alphabet = Technology('alphabet', [], 0)
        ceremonial_burial = Technology('ceremonial_burial', [], 0)
        pottery = Technology('pottery', [], 0)
        masonry = Technology('masonry', [], 0)
        horseback_riding = Technology('horseback_riding', [], 0)
        bronze_working = Technology('bronze_working', [], 0)
        warrior_code = Technology('warrior_code', [], 0)
        writing = Technology('writing', [alphabet], 0)
        code_of_laws = Technology('code_of_laws', [alphabet], 0)
        mysticism = Technology('mysticism', [ceremonial_burial], 0)
        mathematics = Technology('mathematics', [alphabet, masonry], 0)
        map_making = Technology('map_making', [alphabet], 0)
        polytheism = Technology('polytheism', [ceremonial_burial, horseback_riding], 0)
        the_wheel = Technology('the_wheel', [horseback_riding], 0)
        currency = Technology('currency', [bronze_working], 0)
        iron_working = Technology('iron_working', [bronze_working, warrior_code], 0)
        literacy = Technology('literacy', [writing, code_of_laws], 0)
        trade = Technology('trade', [code_of_laws, currency], 0)
        monarchy = Technology('monarchy', [code_of_laws, ceremonial_burial], 0)
        astronomy = Technology('astronomy', [mysticism, mathematics], 0)
        seafaring = Technology('seafaring', [map_making, pottery], 0)
        construction = Technology('construction', [masonry, currency], 0)
        the_republic = Technology('the_republic', [code_of_laws, literacy], 0)
        philosophy = Technology('philosophy', [literacy, mysticism], 0)
        navigation = Technology('navigation', [astronomy, seafaring], 0)
        engineering = Technology('engineering', [the_wheel, construction], 0)
        feudalism = Technology('feudalism', [monarchy, warrior_code], 0)
        bridge_building = Technology('bridge_building', [construction, iron_working], 0)
        banking = Technology('banking', [the_republic, trade], 0)
        medicine = Technology('medicine', [trade, philosophy], 0)
        university = Technology('university', [philosophy, mathematics], 0)
        physics = Technology('physics', [literacy, navigation], 0)
        monotheism = Technology('monotheism', [philosophy, polytheism], 0)
        invention = Technology('invention', [literacy, engineering], 0)
        chivalry = Technology('chivalry', [feudalism, horseback_riding], 0)

        self.techs = {
            'alphabet': alphabet,
            'ceremonial_burial': ceremonial_burial,
            'pottery': pottery,
            'masonry': masonry,
            'horseback_riding': horseback_riding,
            'bronze_working': bronze_working,
            'warrior_code': warrior_code,
            'writing': writing,
            'code_of_laws': code_of_laws,
            'mysticism': mysticism,
            'mathematics': mathematics,
            'map_making': map_making,
            'polytheism': polytheism,
            'the_wheel': the_wheel,
            'currency': currency,
            'iron_working': iron_working,
            'literacy': literacy,
            'trade': trade,
            'monarchy': monarchy,
            'astronomy': astronomy,
            'seafaring': seafaring,
            'construction': construction,
            'the_republic': the_republic,
            'philosophy': philosophy,
            'navigation': navigation,
            'engineering': engineering,
            'feudalism': feudalism,
            'bridge_building': bridge_building,
            'banking': banking,
            'medicine': medicine,
            'university': university,
            'physics': physics,
            'monotheism': monotheism,
            'invention': invention,
            'chivalry': chivalry,
        }

        self.currently_researching = None

    def get_researchable(self):
        """
        Get a list of researchable technologies

        :return: A list of researchable technologies
        """
        researchable = []
        for tech in self.techs.values():
            if all(map(lambda req: req.researched, tech.requirements)):
                researchable.append(tech)
        return researchable

    def research(self, tech):  # TODO check, also duplicated logic.
        """
        Research a technology

        :param tech: The technology
        """
        if 'str' in str(type(tech)):
            if tech not in self.techs:
                raise ValueError(f'{tech} does not exist')
            tech = self.techs[tech]

        if self.techs[tech].researched:
            raise ValueError(f'{tech.name} already researched')

        if tech not in self.get_researchable():
            raise ValueError(f'requirements not met for {tech.name}')

        self.techs[tech].researched = True

    def add_research_progress(self, progress):  # TODO Possibly extraneous, probably just get this from packets
        """
        Progress an ongoing research

        :param progress: Progress
        """
        self.techs[self.currently_researching] += progress
        if self.techs[self.currently_researching].progress >= self.techs[self.currently_researching].cost:
            accum_progress = (self.techs[self.currently_researching].progress -
                              self.techs[self.currently_researching].cost)  # TODO needed?
            self.techs[self.currently_researching].researched = True
            self.currently_researching = None
            # TODO find out if we accumulate the unused progress and apply it to the next researched item?

    def is_busy(self):
        return self.currently_researching is not None


# TODO make proper unit test
if __name__ == '__test__':
    tree = TechnologyTree()
    print(*tree.get_researchable(), sep=', ', end='\n\n')

    # research by technology object
    tree.research(tree.get_researchable()[0])
    print(*tree.get_researchable(), sep=', ', end='\n\n')

    # research by name
    # should unlock seafaring
    tree.research('pottery')
    tree.research('map_making')
    print(*tree.get_researchable(), sep=', ', end='\n\n')

    # invalid research: requirements not met
    try:
        tree.research('construction')
    except ValueError as e:
        print(e)
    print()

    # invalid research: already researched
    try:
        tree.research('alphabet')
    except ValueError as e:
        print(e)
    print()

    # invalid research: does not exist
    try:
        tree.research('rocket_science')
    except ValueError as e:
        print(e)
    print()
